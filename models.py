import torch
from torch.functional import F
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, BatchNorm, JumpingKnowledge
import torch.nn as nn

# base GNN class
class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        pass

    def forward(self, x, edge_index):
        pass
    
class GCN(GNN):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index):
        # node emebdding creation (embedding = h)
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = self.conv3(h, edge_index)

        # applying a classifier
        x = h
        return x, h    

class GAT(GNN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True)
        self.conv2 = GATConv(hidden_channels * num_heads, out_channels, heads=num_heads)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        
        x = h
        return x, h  


class GIN(GNN):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GIN, self).__init__()

        # Define GINConv layers with deeper MLPs
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        ))
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        ))
        self.conv3 = GINConv(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        ))

        # Batch normalization or layer normalization
        self.bn1 = BatchNorm(hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)

        # Learnable layer weights for combining layers
        self.layer_weights = nn.Parameter(torch.ones(3))

        # Final classifier layer
        self.classifier = nn.Linear(hidden_channels, out_channels)
        
        # layer weights for combining layers
        self.linear = nn.Linear(hidden_channels * 3, hidden_channels)

    def forward(self, x, edge_index):
        h1 = self.conv1(x, edge_index)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)

        h2 = self.conv2(h1, edge_index)
        h2 = self.bn2(h2)
        h2 = F.relu(h2)

        h3 = self.conv3(h2, edge_index)
        h3 = self.bn3(h3)

        # Weighted sum of layers
        # h = self.layer_weights[0] * h1 + \
        #     self.layer_weights[1] * h2 + self.layer_weights[2] * h3
            
        h = torch.cat([h1, h2, h3], dim=1)
        h = self.linear(h)  # Apply a linear transformation after concatenation

        # Apply dropout
        h = self.dropout(h)

        # Apply final linear layer for classification
        x = self.classifier(h)

        return x, h
    

class GraphSAGE(GNN):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr='max')
        self.conv3 = SAGEConv(hidden_channels, hidden_channels, aggr='lstm')

        # Jumping Knowledge Network
        self.jump_knowledge = JumpingKnowledge(
            mode='lstm', channels=out_channels, num_layers=3)

        # self.fc_layers = nn.Sequential(
        #     nn.Linear(hidden_channels * 3, hidden_channels),
        #     nn.ReLU(),
        #     nn.Linear(hidden_channels, out_channels)
        # )

    def forward(self, x, edge_index):
        # First layer (mean aggregation)
        h1 = self.conv1(x, edge_index)
        h1 = F.relu(h1)

        # Second layer (max-pooling aggregation)
        h2 = self.conv2(h1, edge_index)
        h2 = F.relu(h2)

        # Third layer (LSTM aggregation)
        h3 = self.conv3(h2, edge_index)

        # Jumping Knowledge: concatenate information from all layers
        h = self.jump_knowledge([h1, h2, h3])

        # Apply classifier
        # x = self.fc_layers(h)
        x = h
        return x, h
