import torch
import torch.nn as nn
import torch.nn.functional as F
from .EncoderRNN import EncoderRNN
from .DecoderRNN import DecoderRNN
import pickle
import numpy as np

class Seq2seq(nn.Module):
	""" Standard sequence-to-sequence architecture with configurable encoder
	and decoder.

	Args:
		encoder (EncoderRNN): object of EncoderRNN
		decoder (DecoderRNN): object of DecoderRNN
		decode_function (func, optional): function to generate symbols from output hidden states (default: F.log_softmax)

	Inputs: input_variable, input_lengths, target_variable, teacher_forcing_ratio
		- **input_variable** (list, option): list of sequences, whose length is the batch size and within which
		  each sequence is a list of token IDs. This information is forwarded to the encoder.
		- **input_lengths** (list of int, optional): A list that contains the lengths of sequences
			in the mini-batch, it must be provided when using variable length RNN (default: `None`)
		- **target_variable** (list, optional): list of sequences, whose length is the batch size and within which
		  each sequence is a list of token IDs. This information is forwarded to the decoder.
		- **teacher_forcing_ratio** (int, optional): The probability that teacher forcing will be used. A random number
		  is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
		  teacher forcing would be used (default is 0)

	Outputs: decoder_outputs, decoder_hidden, ret_dict
		- **decoder_outputs** (batch): batch-length list of tensors with size (max_length, hidden_size) containing the
		  outputs of the decoder.
		- **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
		  state of the decoder.
		- **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
		  representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
		  predicted token IDs, *KEY_INPUT* : target outputs if provided for decoding, *KEY_ATTN_SCORE* : list of
		  sequences, where each list is of attention weights }.

	"""

	def __init__(self, embedding=None, wordDict=None, hidden_size=300, style_size=100, input_dropout_p=0, max_len=100, dropout_p=0, n_layers=1, bidirectional=False, rnn_cell='gru', decode_function=F.log_softmax):
		super(Seq2seq, self).__init__()
		print('net...')
		if embedding==None:
			print('no embedding given. please try again')
			exit(0)
		embedding = torch.FloatTensor(np.load(embedding))
		vocab_size = len(embedding)
		sos_id = 2
		eos_id = 3
		self.encoder = EncoderRNN(vocab_size, max_len, hidden_size, 
				input_dropout_p=input_dropout_p, dropout_p=dropout_p, n_layers=n_layers, bidirectional=bidirectional, rnn_cell=rnn_cell, variable_lengths=True,
				embedding=embedding, update_embedding=False)
		self.style_emb = nn.Embedding(7,style_size)
		self.decoder = DecoderRNN(vocab_size, max_len, int((hidden_size+style_size)*(bidirectional+1)), sos_id, eos_id, n_layers=n_layers, rnn_cell=rnn_cell, bidirectional=bidirectional, 
				input_dropout_p=input_dropout_p, dropout_p=dropout_p, use_attention=False, embedding=embedding, update_embedding=False)
		self.decode_function = decode_function

	def flatten_parameters(self):
		self.encoder.rnn.flatten_parameters()
		self.decoder.rnn.flatten_parameters()

	def forward(self, inputs, target_variable=None,
				teacher_forcing_ratio=0):
		tf_ratio = teacher_forcing_ratio if self.training else 0
		encoder_outputs, encoder_hidden = self.encoder(inputs['question'], inputs['qLengths'])
		style_embedding = self.style_emb(inputs['labels_r'])
		result = self.decoder(inputs=[inputs['response'],inputs['rLengths']],#target_variable,
							  style_embd=style_embedding,
							  encoder_hidden=encoder_hidden, #encoder_hidden0,
							  encoder_outputs=encoder_outputs,
							  function=self.decode_function,
							  teacher_forcing_ratio=tf_ratio,
							  outputs_maxlen=max(inputs['rLengths']))
		return result



class Criterion(nn.Module):
	"""docstring for Criterion"""
	def __init__(self, config):
		super(Criterion, self).__init__()
		print('crit...')
		self.celoss = nn.CrossEntropyLoss()

	def LanguageModelLoss(self):
		pass

	def ReconstructLoss(self):
		pass

	def forward(self, outputs, inputs):
		labels = inputs['sentence']
		lengths = inputs['st_inp_lengths']
		decoder_outputs = outputs[0]
		decoder_outputs = torch.cat([torch.tensor(k).unsqueeze(1) for k in decoder_outputs],1) #[batch, seqlength, vocabsize]

		batchSize = len(labels)
		loss = 0
		for i in range(batchSize):
			wordLogPs = decoder_outputs[i][:lengths[i]-1]
			gtWdIndices = labels[i][1:lengths[i]]
			loss += self.celoss(wordLogPs, gtWdIndices)
			# loss += - torch.sum(torch.gather(wordLogPs,1,gtWdIndices.unsqueeze(1)))/float(lengths[i]-1)
		loss = loss/batchSize
		return loss
			


		