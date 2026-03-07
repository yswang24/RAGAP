import torch
from torch_geometric.nn import GATv2Conv

def test_gat():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on: {device}")
    
    # 模拟 10 个节点，每个节点 16 维特征
    x = torch.randn(10, 16).to(device)
    # 模拟全连接图 (所有节点互连)
    edge_index = torch.cartesian_prod(torch.arange(10), torch.arange(10)).T.to(device)
    
    # 定义 GAT 层
    conv = GATv2Conv(16, 16, edge_dim=1).to(device)
    
    # 模拟边特征
    edge_attr = torch.randn(edge_index.size(1), 1).to(device)
    
    # 前向传播
    out = conv(x, edge_index, edge_attr=edge_attr)
    
    # 检查输出
    print(f"Output Mean: {out.mean().item():.6f}")
    if out.abs().sum() == 0:
        print("❌ FAILED: 输出全为 0，环境依然有问题！")
    else:
        print("✅ SUCCESS: GAT 算子工作正常！数值非零。")

if __name__ == "__main__":
    test_gat()