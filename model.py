import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        #have to give how many words in the vocab (vocab_size), and the dimension of the model (d)

    def forward(self, x):
        return self.embedding(x) . math.sqrt(self.d_model)    # ""In the embedding layers, we multiply those weights by √dmodel."" from the paper where this line comes from
    
    #next module we are gonna build is the positional encoding part 

"""
Our org sentence gets mapped to the embeddigns layer
we wanna convey the info abt the position of each word inside the sentence this i done by adding another vector of the same size by a formula
this tells the model - this word occupies this pos in the sentence"""
class PostionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        #dmodel fo the size of vector, seq lgt for the max lgt of sentence and we want to create one vector for each position, 
        # we also have to give dropout to mak the model less overfit
        super().__init__
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout) 
        #now we build it -- shape of matrix is sequence lgt to dmodel (ie row, col form ==  seq_len,d_model) ? paper check
        #createe a matrix of shape seq_len, d_model IMP
        pe = torch.zeros(seq_len, d_model)
        #use formula to create positional encoding == pe from the paper
        #the sine and cosine fucntions  but we are going to change them a little bit -- use a log space to stabilize it and make it easier too
        # refer gfg for these formulas
        #Even-indexed dimensions: PE(p, 2i) = \sin\left(\frac{p}{10000^{\frac{2i}{d_{\text{model}}}}}\right)
        #Odd-indexed dimensions: PE(p, 2i+1) = \cos\left(\frac{p}{10000^{\frac{2i}{d_{\text{model}}}}}\right)

        #create a vector for the above matrix from 0 to seq_len
        position = torch.arrange(0, seq_len, dtype = torch.float).unsqueeze(1) #what we've just done is create a tensor of seq_len = 1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # this is the denomiantor for the two formulaes -- sine and cosine formulaes
        #i dont understand it well eithr tbh atp but im using it
        #sine is for even -- cosine for odd 
        #aaply sine even
        pe[:, 0::2] = torch.sin(position * div_term)
        #cosine
        pe[:, 1::2] = torch.cosine(position * div_term)


        #we need to add batch dim to apply to full sentence, current shape is seq_len, d_model  BUT we are goign to have a batch of sentences so we need to code for it
        #new dim to pe

        pe = pe.unsqueeze(0) #(1, seq_len, d_model)

        #now we can register the tensor to the buffer
        # when we have a tensor tht we want to keep inside the module, not as  a learned parameter but want to be saved when we save the file so we save it as a buffer

        self.register_buffer('pe', pe)

    #implement forward method, we need to add pe for every word
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False)# we hv to tell the model, that dont learn pe since theyre fixed -- req_grad does tht


# we will do the easiest one next -- we are building the encoder -- next comes the layer normalization -- calc mean and var for each value with each fo the other values
# theres a mult and additive components so that the model changes it
class LayerNormalzation(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
    #eps is  avery small no, needed coz -- addded to the deno in the mean formula to make sure we dont end up dividing by 0 #refer formula for layer normaalization 
    #-- for numerical stbaility we use it as well
        super().__init__
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) #mult component -- learnable too
        self.bias = nn.Parameter(torch.zeros(1)) #added

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim = True) #chatgpt error corrrection, uuslly the mean cancels the dim its applie but we keep it
        std = x.std(sim=-1, keepdim = True) #std deviation
        #now we apply the formulaes
        return self.alpha * (x - mean / (std+self.eps)) + self.bias
            # Mean and standard deviation formulas used in layer normalization:
            # Mean: mean(x) = (1/N) * sum(x_i) for i in 1 to N
            # Standard Deviation: std(x) = sqrt((1/N) * sum((x_i - mean(x))^2)) for i in 1 to N
            # Layer Normalization: 
            #  normalized_x = (x - mean(x)) / (std(x) + epsilon) 
            # where epsilon is a small constant for numerical stability.


#next layer is the feed forward layer, in the paper theres two matrices - W1 * x  RELU  * W2 || W1 is from d_model to dff || W2 is from dff to d_model || d_model=512, dff=2048
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2
        #we have B1 and B2 bcoz bias is true and its defining it by def -- pytorch feature -- ref docu pytorch
    
    def forward(self, x):
        # input tensor which has dim of (Batch, seq_len, d_model) --> convet it using linear 1 to (Bathc, seq_len, d_ff) --> apply linear2 for (btach, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
        #3.3 pg 5 in the paper theres a finction for this -- refer from there

#next block is the multi head attention
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout:float) -> None:
        super().__init__()
        self.d_model =  d_model
        self.h = h
        #mk sure h is divisible by h
        assert d_model % h == 0, "d_model aint divisible by the number of heads"
        #model = 512, h is the number of heads
        #d_K is the d_model divided by h -- given in the paper
        self.d_k = d_model // h
        #query, key and vlue matrices have to multp -- refer paper for multi head attention mech
        self.w_q = nn.Linear(d_model, d_model) #Wq
        self.w_k = nn.Linear(d_model, d_model) #Wk
        self.w_v = nn.Linear(d_model, d_model) #Wv

        self.w_o = nn.Linear(d_model, d_model) #Wo


        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

    #now we implment the mask as well -- siccnce we want some words to not interact with otehrs -- particularly here we dont want old words to know abt future words 
    #we want them to know abt prev words but not future words
    #so we repalce their attention score to epsilon -- eps or osmething vv small -- so that their softmax will become 0 since its e^x where x= -infinity ==> e= vv close to 0
    # therfore we hide the attention for it
    def forward(self, q,k,v, mask):
        query = self.w_q(q) # dimension == (batch, seq_len, d_model)-->(batch, seq_len, d_model)
        key = self.w_k(k) #dimention == same as above
        value = self.w_v(v) # dim == same as above
        #we want divide query key value into diff heads -- into small matrices -- used view method here to keep the batch dimension -- we want to keep dimension and split 
        # embeddings
        #d_k * h == d_model # we want h to be teh second dimension -- for better visuals
        #dimensions == (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query =  query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2) 
        #do same for key and value
        key =  key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2) #dimensions == same as above -- same as query 
        value =  value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2) #dimensions == same as above






        





        

        


   