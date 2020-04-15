# GCN library 
This is heavily borrowed from our [DeepGCNs repo](https://github.com/lightaime/deep_gcns_torch).

There are two folders. 

`sparse` supports data with shape is  `N x feature_size` and there should be variable called _batch_ indicating which batch each node belongs to. 

`dense` supports data with shape is `Batch_size x feature_size x nodes_num x 1`.
