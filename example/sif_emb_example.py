from fse import Vectors, Average, IndexedList
vecs = Vectors.from_pretrained("glove-wiki-gigaword-50")
model = Average(vecs)
sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
model.train(IndexedList(sentences))

model.sv.similarity(0,1)