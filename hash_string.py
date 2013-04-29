#Define a function, hash_string,
#that takes as inputs a keyword
#(string) and a number of buckets,
#and outputs a number representing
#the bucket for that keyword.

BUCKETS=1000000000

def hash_string(keyword,buckets=BUCKETS):
    number = 0
    for letter in keyword:
        number = number + ord(letter) #turn the letter into a number
    return number%buckets
       
def hash_merge(hash1, hash2, buckets=BUCKETS):
    return (hash_string(u' ')+hash1+hash2) % buckets
     
#print hash_string('a',12)# => 1
#print hash_string('b',12)# => 2
#print hash_string('a',13)# => 6
#print hash_string('au',12)# => 10
#print hash_string('udacity',12) # => 11
