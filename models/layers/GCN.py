import torch

class GraphCN(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        # self.batch_nomal = torch.nn.BatchNorm2d(out_features)
        # self.v = torch.nn.Parameter(torch.rand(attention_hidden_size))
        # stdv = 1. / math.sqrt(self.v.size(0))
        # self.weight.data.uniform_(-1.0, 1.0)
        # print('self.weight = ', self.weight)
        self.weight = torch.nn.Parameter(torch.empty(size=(in_features, out_features)))
        torch.nn.init.xavier_uniform_(self.weight.data, gain=1.414)
    
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
            torch.nn.init.xavier_uniform_(self.bias.data, gain=1.414)
            # print('self.bias = ', self.bias)
            # self.bias.data.uniform_(-1.0, 1.0)
        else:
            self.register_parameter('bias', None)
        

    def forward(self, text, adj):
        # print('text = ', text.float())
        # print('self.weight = ', self.weight)
        # self.weight = self.batch_nomal(self.weight)
        hidden = torch.matmul(text.float(), self.weight)
        # print('hidden.type() = ', hidden.type())
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        # print('denom.size( = ', denom.size())
        # print('adj.type() = ', adj.type())
        output = torch.matmul(adj.float(), hidden) / denom.float()
        # print('output = ', output)
        if self.bias is not None:
            # print('self.bias= ', self.bias.size())
            return output + self.bias
        else:
            return output
