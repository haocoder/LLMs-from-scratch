import torch
import torch.nn.functional as F
import numpy as np

def demonstrate_attention_mechanism():
    """
    演示为什么需要 Q、K、V 三个矩阵
    """
    
    # 示例：句子 "I love machine learning"
    # 假设每个词的嵌入维度是 4
    seq_len = 4
    embed_dim = 4
    
    # 模拟词嵌入
    embeddings = torch.tensor([
        [1.0, 0.5, 0.2, 0.1],  # I
        [0.3, 1.0, 0.4, 0.2],  # love  
        [0.1, 0.3, 1.0, 0.8],  # machine
        [0.2, 0.1, 0.7, 1.0]   # learning
    ], dtype=torch.float32)
    
    print("原始词嵌入:")
    print(embeddings)
    print()
    
    # 1. 如果只用两个矩阵（Q 和 K）
    print("=== 只用两个矩阵的情况 ===")
    W_q = torch.randn(embed_dim, embed_dim)
    W_k = torch.randn(embed_dim, embed_dim)
    
    Q = embeddings @ W_q
    K = embeddings @ W_k
    
    # 计算注意力分数
    attention_scores = Q @ K.T
    attention_weights = F.softmax(attention_scores / np.sqrt(embed_dim), dim=-1)
    
    # 输出 = 注意力权重 × K
    output_two_matrices = attention_weights @ K
    
    print("注意力权重:")
    print(attention_weights)
    print()
    print("输出（只用Q和K）:")
    print(output_two_matrices)
    print()
    
    # 2. 使用三个矩阵（Q、K、V）
    print("=== 使用三个矩阵的情况 ===")
    W_v = torch.randn(embed_dim, embed_dim)
    V = embeddings @ W_v
    
    # 输出 = 注意力权重 × V
    output_three_matrices = attention_weights @ V
    
    print("注意力权重（相同）:")
    print(attention_weights)
    print()
    print("输出（使用Q、K、V）:")
    print(output_three_matrices)
    print()
    
    # 3. 展示差异
    print("=== 两种方法的差异 ===")
    difference = output_three_matrices - output_two_matrices
    print("差异:")
    print(difference)
    print(f"差异的范数: {torch.norm(difference):.4f}")
    
    return {
        'two_matrices': output_two_matrices,
        'three_matrices': output_three_matrices,
        'attention_weights': attention_weights
    }

def explain_why_three_matrices():
    """
    解释为什么需要三个矩阵的理论原因
    """
    print("=== 为什么需要 Q、K、V 三个矩阵？ ===\n")
    
    print("1. 分离关注机制和内容表示")
    print("   - Q: 决定'我想关注什么'")
    print("   - K: 决定'我如何被关注'") 
    print("   - V: 决定'我的实际内容是什么'")
    print()
    
    print("2. 更灵活的表示学习")
    print("   - 同一个词在不同上下文中可能有不同的关注模式")
    print("   - 但内容表示可能保持相对稳定")
    print()
    
    print("3. 数学上的优势")
    print("   - 可以学习到更丰富的表示空间")
    print("   - 避免了表示冲突")
    print("   - 提高了模型的表达能力")
    print()
    
    print("4. 实际例子:")
    print("   句子: 'The bank by the river'")
    print("   - 'bank' 作为名词时，K 矩阵让它容易被'river'关注")
    print("   - 'bank' 的 V 矩阵包含'金融机构'或'河岸'的含义")
    print("   - 上下文决定最终选择哪个含义")

if __name__ == "__main__":
    results = demonstrate_attention_mechanism()
    print("\n" + "="*50 + "\n")
    explain_why_three_matrices()
