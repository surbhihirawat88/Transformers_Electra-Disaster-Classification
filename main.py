from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import ElectraTokenizer, ElectraForSequenceClassification, AdamW
from flask import Flask, request, render_template
import numpy as np
import torch


if torch.cuda.is_available():
	device = torch.device("cuda")
	print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
	print('No GPU available, using the CPU instead.')
	device = torch.device("cpu")


def prediction(text):
	tokenizer1 = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
	PATH = "/mnt/01D557900A25E360/Study/projects/ELECTRA_FINAL-20201210T023739Z-001/ELECTRA_FINAL/state_electra_final_model.pt"
	model1 = torch.load(PATH, map_location='cpu')
	line_tokenized = tokenizer1.batch_encode_plus(text,max_length=64,add_special_tokens=True, return_attention_mask=True,pad_to_max_length=True,truncation=True)
	input_ids1 = line_tokenized["input_ids"]
	attention_masks1 = line_tokenized["attention_mask"]
	prediction_inputs1 = torch.tensor(input_ids1)
	prediction_masks1 = torch.tensor(attention_masks1)
	prediction_data1 = TensorDataset(prediction_inputs1, prediction_masks1)
	prediction_sampler1 = SequentialSampler(prediction_data1)
	prediction_dataloader1 = DataLoader(prediction_data1, sampler=prediction_sampler1, batch_size=1)
	
	predictions = []
	
	for batch in prediction_dataloader1:
		batch = tuple(t.to('cpu') for t in batch)
		b_input_ids, b_input_mask = batch
		with torch.no_grad():
			
			outputs1 = model1(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
			print(outputs1)
		logits1 = outputs1[0]
		logits1 = logits1.detach().cpu().numpy()
		predictions.append(logits1)
		flat_predictions = [item for sublist in predictions for item in sublist]
		flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
	
	return flat_predictions




app = Flask(__name__)  # initializing a flask app


@app.route('/', methods=['GET'])  # route to display the home page
def homepage():
	return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])  # route to show the predictions in a web UI
def index():
	if request.method == 'POST':
		try:
			print(request.form['line'])
			message = request.form['line']
			data = [message]
			vect=prediction(data)
			return render_template('results.html', data=vect)
		
		except Exception as e:
			print('The Exception message is: ', e)
			# return 'something is wrong'
			raise e
			# return render_template('results.html')
	else:
		return render_template('results.html')


if __name__ == "__main__":
	# app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(debug=True)  # running the app


