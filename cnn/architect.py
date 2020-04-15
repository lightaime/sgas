import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  def _compute_unrolled_model(self, input, target, eta, network_optimizer):
    loss = self.model._loss(input, target)
    theta = _concat(self.model.parameters()).data
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    xs = torch.autograd.grad(loss, self.model.parameters(), allow_unused=True)
    xs_new = []
    for x, p in zip(xs, self.model.parameters()):
      if x is None:
        x = torch.zeros_like(p)
      xs_new.append(x)
    dtheta = _concat(xs_new).data + self.network_weight_decay*theta
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
    return unrolled_model

  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
    else:
        self._backward_step(input_valid, target_valid)
    for weights in self.model.arch_parameters():
      if not weights.requires_grad:
        weights.grad = None
    self.optimizer.step()

  def _backward_step(self, input_valid, target_valid):
    loss = self.model._loss(input_valid, target_valid)
    loss.backward()

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
    unrolled_loss = unrolled_model._loss(input_valid, target_valid)

    unrolled_loss.backward()
    dalpha = [v.grad if v.grad is not None else torch.zeros_like(v) for v in unrolled_model.arch_parameters()]
    vector = [v.grad.data if v.grad is not None else torch.zeros_like(v).data for v in unrolled_model.parameters()]
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

  def _construct_model_from_theta(self, theta):
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _hessian_vector_product(self, vector, input, target, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)
    loss = self.model._loss(input, target)
    requires_grad_arch_parameters = [p for p in self.model.arch_parameters() if p.requires_grad]
    req_grads_p = torch.autograd.grad(loss, requires_grad_arch_parameters)
    grads_p = []
    count_p = 0
    for a_p in self.model.arch_parameters():
      if a_p.requires_grad:
        g_p = req_grads_p[count_p]
        count_p = count_p + 1
      else:
        g_p = torch.zeros_like(a_p)
      grads_p.append(g_p)

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)
    loss = self.model._loss(input, target)
    req_grads_n = torch.autograd.grad(loss, requires_grad_arch_parameters)
    grads_n = []
    count_n = 0
    for a_p in self.model.arch_parameters():
      if a_p.requires_grad:
        g_n = req_grads_n[count_n]
        count_n = count_n + 1
      else:
        g_n = torch.zeros_like(a_p)
      grads_n.append(g_n)

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

