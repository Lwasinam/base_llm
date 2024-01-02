
##Implementation of tranformer from scratch, this implememtation was inspired by Umar Jamir

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super(InputEmbeddings, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)


    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)


        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model) 


class PositionEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model:int, batch: int) -> None:
        super(PositionEncoding, self).__init__()
        # self.seq_len = seq_len
        # self.d_model = d_model
        # self.batch = batch
        self.dropout = nn.Dropout(p=0.1)
    
        ##initialize the positional encoding with zeros
        positional_encoding = torch.zeros(seq_len, d_model)
     
        ##first path of the equation is postion/scaling factor per dimesnsion
        postion  = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    
        ## this calculates the scaling term per dimension (512)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # div_term = torch.pow(10,  torch.arange(0,self.d_model, 2).float() *-4/self.d_model)
      

        ## this calculates the sin values for even indices
        positional_encoding[:, 0::2] = torch.sin(postion * div_term) 

      
        ## this calculates the cos values for odd indices
        positional_encoding[:, 1::2] = torch.cos(postion * div_term)

        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer('positional_encoding', positional_encoding)
    
    def forward(self, x):  
         x = x + (self.positional_encoding[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
         return self.dropout(x)



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, heads: int) -> None:
        super(MultiHeadAttention,self).__init__()
        self.head = heads
        self.head_dim = d_model // heads
        


        assert d_model % heads == 0, 'cannot divide d_model by heads'

        ## initialize the query, key and value weights 512*512
        self.query_weight = nn.Linear(d_model, d_model, bias=False)
        self.key_weight = nn.Linear(d_model, d_model,bias=False)
        self.value_weight = nn.Linear(d_model, d_model,bias=False)
        self.final_weight  = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(p=0.1)

      
    def self_attention(self,query, key, value, mask,dropout):
        #splitting query, key and value into heads
                #this gives us a dimension of batch, num_heads, seq_len by 64. basically 1 sentence is converted to have 8 parts (heads)
        query = query.view(query.shape[0], query.shape[1],self.head,self.head_dim).transpose(2,1)
        key = key.view(key.shape[0], key.shape[1],self.head,self.head_dim).transpose(2,1)
        value = value.view(value.shape[0], value.shape[1],self.head,self.head_dim).transpose(2,1)
        
        attention = query @ key.transpose(3,2)
        attention = attention / math.sqrt(query.shape[-1])

        if mask is not None:
           attention = attention.masked_fill(mask == 0, -1e9)      
        attention = torch.softmax(attention, dim=-1)      
        if dropout is not None:
            attention = dropout(attention)
        attention_scores =  attention @ value    
       
        return attention_scores.transpose(2,1).contiguous().view(attention_scores.shape[0], -1, self.head_dim * self.head)
      
    def forward(self,query, key, value,mask):

        ## initialize the query, key and value matrices to give us seq_len by 512
        query = self.query_weight(query)
        key = self.key_weight(key)
        value = self.value_weight(value)

        attention = MultiHeadAttention.self_attention(self, query, key, value, mask, self.dropout)
        return self.final_weight(attention) 

class FeedForward(nn.Module):
    def __init__(self,d_model:int, d_ff:int ) -> None:
        super(FeedForward, self).__init__()

        self.fc1 = nn.Linear(d_model, d_ff)  # Fully connected layer 1
        self.dropout = nn.Dropout(p=0.1)  # Dropout layer
        self.fc2 = nn.Linear(d_ff, d_model)  # Fully connected layer 2
     
    
    def forward(self,x ):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))  

class ProjectionLayer(nn.Module):
    def __init__(self, d_model:int, vocab_size:int) :
        super(ProjectionLayer, self).__init__()
        self.fc = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        x = self.fc(x)
        return torch.log_softmax(x, dim=-1)   

class EncoderBlock(nn.Module):
    def __init__(self, d_model:int, head:int, d_ff:int) -> None:
        super(EncoderBlock, self).__init__()    
        self.multiheadattention = MultiHeadAttention(d_model,head)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=0.1)
        self.feedforward = FeedForward(d_model, d_ff)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=0.1)

    def forward(self, x, src_mask):
       # Self-attention block
        norm = self.layer_norm1(x)
        attention = self.multiheadattention(norm, norm, norm, src_mask)
        x = (x + self.dropout1(attention))

        # Feedforward block
        norm2 = self.layer_norm2(x)
        ff = self.feedforward(x)
        return x + self.dropout2(ff)     

class Encoder(nn.Module):
    def __init__(self, number_of_block:int, d_model:int, head:int, d_ff:int) -> None:
        super(Encoder, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        
        # Use nn.ModuleList to store the EncoderBlock instances
        self.encoders = nn.ModuleList([EncoderBlock(d_model, head, d_ff) 
                                       for _ in range(number_of_block)])

    def forward(self, x, src_mask):
        for encoder_block in self.encoders:
            x = encoder_block(x, src_mask)
        return self.norm(x)   
   
class DecoderBlock(nn.Module):
    def __init__(self, d_model:int, head:int, d_ff:int) -> None:
        super(DecoderBlock, self).__init__()
        self.head_dim = d_model // head
        
        self.multiheadattention = MultiHeadAttention(d_model, head)
        self.crossattention = MultiHeadAttention(d_model, head)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=0.1)
        self.feedforward = FeedForward(d_model,d_ff)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.layer_norm4 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.1)
    def forward(self, x, tgt_mask):
        #Self-attention block
        norm = self.layer_norm1(x)
        attention = self.multiheadattention(norm, norm, norm, tgt_mask)
        x = (x + self.dropout1(attention))
    
        # # Cross-attention block
        # norm2 = self.layer_norm2(x)    
        # cross_attention = self.crossattention(norm, encoder_output, encoder_output, src_mask)
        # x = (x + self.dropout2(cross_attention))
   
        # Feedforward block  
        norm3  = self.layer_norm3(x)
        ff = self.feedforward(norm3)
        return x + self.dropout3(ff)   


class Decoder(nn.Module):
    def __init__(self, number_of_block:int,d_model:int, head:int, d_ff:int) -> None:
        super(Decoder, self).__init__()
        self.norm = nn.LayerNorm(d_model) 
        self.decoders = nn.ModuleList([DecoderBlock(d_model, head, d_ff) 
                                       for _ in range(number_of_block)])

    def forward(self, x, tgt_mask):
        for decoder_block in self.decoders:
            x = decoder_block(x, tgt_mask)
        return self.norm(x)    


class Transformer(nn.Module):
    def __init__(self, seq_len:int, batch:int, d_model:int,target_vocab_size:int, source_vocab_size:int, head: int = 8, d_ff: int =  2048, number_of_block: int = 6) -> None:
        super(Transformer, self).__init__()
        # self.encoder = Encoder(number_of_block,d_model, head, d_ff )
        self.decoder = Decoder(number_of_block, d_model, head, d_ff )
        self.projection = ProjectionLayer(d_model, target_vocab_size)
        self.source_embedding = InputEmbeddings(d_model,source_vocab_size )
        self.target_embedding = InputEmbeddings(d_model,target_vocab_size)
        self.positional_encoding = PositionEncoding(seq_len, d_model, batch)

   
    # def encode(self,x, src_mask):
    #     x = self.source_embedding(x)
    #     x = self.positional_encoding(x)
    #     return self.encoder(x, src_mask)
       
    def decode(self,x, tgt_mask):
        x = self.target_embedding(x)
        x = self.positional_encoding(x)
        return self.decoder(x, tgt_mask,)
        
    def project(self, x):
        return self.projection(x)
        


def build_transformer(seq_len, batch, target_vocab_size, source_vocab_size,  d_model)-> Transformer:
    

    transformer = Transformer(seq_len, batch,  d_model,  target_vocab_size, source_vocab_size )

      #Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer         


