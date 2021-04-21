
# Stance detection 

Stance  detection  has  been  defined  as  automatically  detecting  whether  the  author  of apiece of text is in favor of the given target or against it. [1] 
during my 2.5 month internship at icog-labs i developed a neural network architecture for stance detection 
the neural network have 2 self attention layers relating different positions of a single sequence in order to compute a representation of the same sequence [2] body and head representations
pass through the self-attention layers the output of the attention representation will be concatenated and feed it to a fully-connected layer <br>
This is also a part of fake news detection competition 

## Installation 

```bash 
   pip install -r requirements.txt
```

## Usage

run ./notebook/Stance Detaction.ipynb

## Docker

docker build -t stance .
docker run -p 8888:8888 stance

## References

[1] https://arxiv.org/pdf/1701.00504 <br>
[2] https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html
