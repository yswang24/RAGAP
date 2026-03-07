from transformers import BertTokenizer, BertModel

model_path = "./DNA_bert_6"  # 切换成你下载的模型路径

# 加载 tokenizer 和模型
tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
model = BertModel.from_pretrained(model_path)

# 简单测试输入
seq = "ATCGTACGATCG"
k = 6
tokens = [seq[i:i+k] for i in range(len(seq)-k+1)]
print("K-mers:", tokens)

# 编码
inputs = tokenizer(" ".join(tokens), return_tensors="pt")
outputs = model(**inputs)

print("Embedding shape:", outputs.last_hidden_state.shape)
