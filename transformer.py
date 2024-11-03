import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
block_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(PositionalEncoding, self).__init__()
        #initialize word and position embeddings
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)

    def forward(self, x):
        #generate position indices and apply embeddings
        positions = torch.arange(0, x.size(1)).unsqueeze(0).to(device)
        return self.word_embedding(x) + self.position_embedding(positions)


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_per_head = embedding_dim // num_heads

        #apply linear transformations for queries, keys and values
        self.q_linear = nn.Linear(embedding_dim, embedding_dim)
        self.k_linear = nn.Linear(embedding_dim, embedding_dim)
        self.v_linear = nn.Linear(embedding_dim, embedding_dim)
        self.out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        batch_size = x.size(0)
        
        # calcualte QKV for all heads
        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        
        # compute scaled attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.dim_per_head, dtype=torch.float32))
        attn = F.softmax(scores, dim=-1)
        
        #apply attention to values
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.dim_per_head)

        return self.out(context), attn


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(TransformerEncoderLayer, self).__init__()
        # initialize multi head attention layer
        self.attention = MultiHeadAttention(embedding_dim, num_heads)
        
        #initialize normalization and feed-forward layers
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, 2 * embedding_dim),
            nn.ReLU(),
            nn.Linear(2 * embedding_dim, embedding_dim)
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        
        #self attention 
        attn_out, attn = self.attention(x)
        x = self.norm1(x + attn_out)
        
        #feed forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x, attn


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers):
        super(TransformerEncoder, self).__init__()
        #embededing with positional encoding
        self.embedding = PositionalEncoding(vocab_size, embedding_dim)
        #setting up encoder layers
        self.layers = nn.ModuleList([TransformerEncoderLayer(embedding_dim, num_heads) for _ in range(num_layers)])

    def forward(self, x):
        #apply embedding and pass through each layers
        x = self.embedding(x)
        attn_maps = []
        for layer in self.layers:
            x, attn = layer(x)
            attn_maps.append(attn)
        return x, attn_maps


class Classifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()
        #set up 2 layers
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class TransformerDecoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim=100):
        super(TransformerDecoderLayer, self).__init__()
        # Masked multi-head attention layer
        self.attention = MaskedMultiHeadAttention(embedding_dim, num_heads)
        
        # Layer normalization and feed-forward layers
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # Self-attention with residual connection
        attn_out, attn = self.attention(x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x, attn

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers):
        super(TransformerDecoder, self).__init__()
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Stack of decoder layers
        self.layers = nn.ModuleList([TransformerDecoderLayer(embedding_dim, num_heads) for _ in range(num_layers)])
        
        # Output layer projecting to vocabulary size
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        # Apply embedding and pass through each decoder layer
        x = self.embedding(x)
        attn_maps = []
        for layer in self.layers:
            x, attn = layer(x)
            attn_maps.append(attn)
        
        # Output logits for each position in sequence
        output = self.output_layer(x)
        
        return output, attn_maps

# Masked Self-Attention to ensure decoder only attends to previous tokens
class MaskedMultiHeadAttention(MultiHeadAttention):
    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        q = self.q_linear(x).view(batch_size, seq_length, self.num_heads, self.dim_per_head).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_length, self.num_heads, self.dim_per_head).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_length, self.num_heads, self.dim_per_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.dim_per_head, dtype=torch.float32)).to(x.device)

        # Apply mask to ensure attention is only applied to past tokens
        mask = torch.tril(torch.ones(seq_length, seq_length)).to(x.device)
        mask = mask.unsqueeze(0).unsqueeze(1)  # Shape (1, 1, seq_length, seq_length)
        scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.dim_per_head)

        return self.out(context), attn
    
### below are implementations for part 3

#longformer for encoder

class LongformerAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, attention_window):
        super(LongformerAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_per_head = embedding_dim // num_heads
        self.attention_window = attention_window

        # Linear transformations for queries, keys, and values
        self.q_linear = nn.Linear(embedding_dim, embedding_dim)
        self.k_linear = nn.Linear(embedding_dim, embedding_dim)
        self.v_linear = nn.Linear(embedding_dim, embedding_dim)
        self.out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Generate Q, K, V for each head
        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)

        # Compute scaled dot-product attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.dim_per_head, dtype=torch.float32))
        
        # Apply local attention mask
        mask = self.create_attention_mask(seq_len).to(x.device)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Compute attention weights
        attn = F.softmax(scores, dim=-1)

        # Compute context vector by applying attention weights to values
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.dim_per_head)

        return self.out(context), attn

    # Create an attention mask to restrict attention to a window around each position
    def create_attention_mask(self, seq_len):
        # Initialize mask with zeros (meaning no attention)
        mask = torch.zeros((seq_len, seq_len), dtype=torch.bool)
        
        # Create a sliding window mask for each token
        for i in range(seq_len):
            start = max(0, i - self.attention_window)  # Determine start of the attention window
            end = min(seq_len, i + self.attention_window + 1)  # Determine end of the attention window
            mask[i, start:end] = 1  # Enable attention within the defined window
        
        return mask

class TransformerEncoderLayer2(nn.Module):
    def __init__(self, embedding_dim, num_heads, attention_window):
        super(TransformerEncoderLayer2, self).__init__()
        self.attention = LongformerAttention(embedding_dim, num_heads, attention_window)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, 2 * embedding_dim),
            nn.ReLU(),
            nn.Linear(2 * embedding_dim, embedding_dim)
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        attn_out, attn = self.attention(x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x, attn


class TransformerEncoder2(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, attention_window=128):
        super(TransformerEncoder2, self).__init__()
        self.embedding = PositionalEncoding(vocab_size, embedding_dim)
        self.layers = nn.ModuleList([TransformerEncoderLayer2(embedding_dim, num_heads, attention_window) for _ in range(num_layers)])

    def forward(self, x):
        x = self.embedding(x)
        attn_maps = []
        for layer in self.layers:
            x, attn = layer(x)
            attn_maps.append(attn)
        return x, attn_maps

    
#longformer implementation for decoder
class LongformerAttention2(nn.Module):
    def __init__(self, embedding_dim, num_heads, attention_window):
        super(LongformerAttention2, self).__init__()
        self.num_heads = num_heads
        self.dim_per_head = embedding_dim // num_heads
        self.attention_window = attention_window

        # Linear transformations for queries, keys, and values
        self.q_linear = nn.Linear(embedding_dim, embedding_dim)
        self.k_linear = nn.Linear(embedding_dim, embedding_dim)
        self.v_linear = nn.Linear(embedding_dim, embedding_dim)
        self.out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # Generate Q, K, V for each head
        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)

        # Compute scaled dot-product attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.dim_per_head, dtype=torch.float32))
        
        # Apply local attention mask
        local_mask = self.create_attention_mask(seq_len).to(x.device)
        scores = scores.masked_fill(local_mask == 0, float('-inf'))

        # Apply additional mask (e.g., causal mask for decoder)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Compute attention weights
        attn = F.softmax(scores, dim=-1)

        # Compute context vector by applying attention weights to values
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.dim_per_head)

        return self.out(context), attn

    # Create an attention mask to restrict attention to a window around each position
    def create_attention_mask(self, seq_len):
        # Initialize mask with zeros (meaning no attention)
        mask = torch.zeros((seq_len, seq_len), dtype=torch.bool)
        
        # Create a sliding window mask for each token
        for i in range(seq_len):
            start = max(0, i - self.attention_window)  # Determine start of the attention window
            end = min(seq_len, i + self.attention_window + 1)  # Determine end of the attention window
            mask[i, start:end] = 1  # Enable attention within the defined window
        
        return mask


class TransformerDecoderLayer2(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim=100, attention_window=50):
        super(TransformerDecoderLayer2, self).__init__()
        self.attention = LongformerAttention2(embedding_dim, num_heads, attention_window)  # Using LongformerAttention for decoder
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x, mask=None):
        attn_out, attn = self.attention(x, mask=mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x, attn


class TransformerDecoder2(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, attention_window=50):
        super(TransformerDecoder2, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.layers = nn.ModuleList([TransformerDecoderLayer2(embedding_dim, num_heads, attention_window=attention_window) for _ in range(num_layers)])
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        batch_size, seq_len, _ = x.size()

        # Create a causal mask to ensure that each position can only attend to previous positions
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool)).to(x.device)
        mask = mask.unsqueeze(0).unsqueeze(1)  # Adjust dimensions for multi-head attention

        attn_maps = []
        for layer in self.layers:
            x, attn = layer(x, mask=mask)
            attn_maps.append(attn)
        output = self.output_layer(x)
        return output, attn_maps