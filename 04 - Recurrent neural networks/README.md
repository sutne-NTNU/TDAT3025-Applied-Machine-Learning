# Recurrent neural networks
## Part A -  Many to many LSTM (Long Short-term Memory)
ta utgangspunkt i [rnn/generate-characters](https://gitlab.com/ntnu-tdat3025/rnn/generate-characters) og tren modellen på
bokstavene “ hello world “. Bruk deretter modellen til å generere 50 bokstaver etter inputen “ h”.

### results
Expanding the example code on these points, training with the phrase `" hello world"`:
```python
char_encodings = [
    [1., 0., 0., 0., 0., 0., 0., 0.],  # ' '
    [0., 1., 0., 0., 0., 0., 0., 0.],  # 'h'
    [0., 0., 1., 0., 0., 0., 0., 0.],  # 'e'
    [0., 0., 0., 1., 0., 0., 0., 0.],  # 'l'
    [0., 0., 0., 0., 1., 0., 0., 0.],  # 'o'
    [0., 0., 0., 0., 0., 1., 0., 0.],  # 'w'
    [0., 0., 0., 0., 0., 0., 1., 0.],  # 'r'
    [0., 0., 0., 0., 0., 0., 0., 1.],  # 'd'
]
index_to_char = [' ', 'h', 'e', 'l', 'o', 'w', 'r', 'd']

x_train = torch.tensor(
    [[char_encodings[0]],  # ' '
     [char_encodings[1]],  # 'h'
     [char_encodings[2]],  # 'e'
     [char_encodings[3]],  # 'l'
     [char_encodings[3]],  # 'l'
     [char_encodings[4]],  # 'o'
     [char_encodings[0]],  # ' '
     [char_encodings[5]],  # 'w'
     [char_encodings[4]],  # 'o'
     [char_encodings[6]],  # 'r'
     [char_encodings[3]],  # 'l'
     [char_encodings[7]],  # 'd'
     ])

y_train = torch.tensor(
    [char_encodings[1],  # 'h'
     char_encodings[2],  # 'e'
     char_encodings[3],  # 'l'
     char_encodings[3],  # 'l'
     char_encodings[4],  # 'o'
     char_encodings[0],  # ' '
     char_encodings[5],  # 'w'
     char_encodings[4],  # 'o'
     char_encodings[6],  # 'r'
     char_encodings[3],  # 'l'
     char_encodings[7],  # 'd'
     char_encodings[0],  # ' '
     ])
``` 
### Result
```console
hooooooooooooooooooooooooooooooooooooooooooooooooooo
hoo                                                 
hll                                                 
hloolooodddddddddddddddddddddddddddddddddddddddddddd
hlllllll                                            
hlooorrddddddddddddddddddddddddddddddddddddddddddddd
hlloooooo                                           
hlllo    d d d d d dd d dd d dd d dd d dd d dd d dd 
hllooooollll   ll   ll    ll    ll    ll    ll    ll
hlloo  rdd d d d d d d d d d d d d d d d d d d d d d
hlllooo rll                                         
hllloorldd                                          
hllloooowrrld                                       
hlll  rdd                                           
hlllo wrrll    dd      d    d    d   d   d   d   d  
hlllo wrll                                          
hlllo world     d                                   
hlllo world    d                                    
hlllo world     rdd    rdd    rdd    rdd    rdd    r
hlllo wrld    d     d     d     d     d     d     d 
hlllo worrld   world   world   world   world   world
hlll world   rdd    dd    dd    dd    dd    dd    dd
hlllo world   wrld   wrld   wrld   wrld   wrld   wrl
hlllo world   wrld   wrld   rld   wrld   wrld   rld 
hlllo world   wrld   wrld   wrld   wrld   wrld   wrl
hlllo world   wrld   wrld   wrld   wrld   wrld   wrl
hlllo world   wrld   wrld   wrld   wrld   wrld   wrl
hlllo world   wrld   wrld   wrld   wrld   wrld   wrl
hlllo world   wrld   wrld   wrld   wrld   wrld   wrl
...
hello world world world world world world world worl
hello world world world world world world world worl
hello world world world world world world world worl
```


## Part B -  Many to one LSTM
tren modellen ulike ord (bruk fortsatt bokstavkoding som i
oppgave a) for emojis, for eksempel  
- “hat ”: 🎩, 
- “rat “: 🐀, 
- “cat “: 🐈, 
- “flat”: 🏠, 
- “matt”: 🧑🏻, 
- “cap “: 🧢, 
- “son ”: 👶🏻   

For å kunne trene i batches er ordene padded med mellomrom på slutten (hvis ordene er mindre enn makslengden).
Test deretter modellen på ord som “rt ” og “rats”, og se hvilken emoji du får.

### Result
I used the following testdata:
```console
"rat " : 🐁
"hat " : 👒
"cat " : 🐈️
"bat " : 🦇
"flat" : 🏡
"matt" : 🤵🏻
"egg " : 🥚
"meat" : 🥩
"ball" : ⚽
```
And in order for me not to go crazy by having to manually typing in `1.` and `0.` and keeping track of the indeces of all the letters i am using a couple functions to do it for me:
```python
# Turns array of letters/emojies into diagonal matrix of 0. and 1.
def encodeChars(arrayOfChars):
    return [[1. if i == j else 0. for j in range(len(arrayOfChars))] for i in range(len(arrayOfChars))]

# Turns the string into list of [char_encoding[x]]
def encodeWord(string):
    return [char_encodings[chars.index(letter)] for letter in string]
```
Below you can see the result after a couple of epochs, with the test-input on the left.
```console
e̲p̲o̲c̲h̲:̲|̲ ̲0̲ ̲ ̲ ̲ ̲5̲ ̲ ̲ ̲ ̲1̲0̲ ̲ ̲ ̲ ̲1̲5̲ ̲ ̲ ̲ ̲2̲0̲ ̲ ̲ ̲ ̲2̲5̲ ̲ ̲ ̲ 
rt    | 🥩   🐁    🐁    🐁    🐁    🐁     
ht    | 👒   🏡    🏡    👒    👒    👒     
c     | 👒   👒    👒    🐈️    🐈️    🐈️    
bt    | 🥩   🏡    🏡    🏡    🦇    🦇     
f     | 👒   🏡    🏡    🏡    🏡    🏡     
tt    | 🥩   🤵🏻    🤵🏻    🤵🏻    🤵🏻    🤵🏻    
g     | 🥚   🥚    🥚    🥚    🥚    🥚     
ea    | 🥩   🥩    🥩    🥩    🥩    🥩     
l     | ⚽   ⚽    ⚽    ⚽    ⚽    ⚽     
```
