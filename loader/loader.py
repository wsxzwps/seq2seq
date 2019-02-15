import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from utils import subset, seq_collate, StyleMarker
import numpy as np 

class CustomDataset(Dataset):
	"""docstring for Dataset"""
	# dataset behave differently when requesting label or unlabel data
	POS = 1
	NEG = 0
	OppStyle = {POS:NEG,NEG:POS}
	def __init__(self, config, datafile, emotion_label_file, forceNoNoise=False): #, wordDictFile): #, labeled=True, needLabel=True):
		super(CustomDataset, self).__init__()
		print('- dataset: '+datafile)
		# self.data = {self.POS:[], self.NEG:[]}
		self.data = self.readData(datafile, emotion_label_file)

		with open(config['wordDict'],"rb") as fp:
			self.wordDict = pickle.load(fp,encoding='latin1')
		self.sos_id = 2 
		self.eos_id = 3
		self.unk_id = 1
		# self.isTrans = config['isTrans']

		# self.sm = StyleMarker(config['selfatt'],self.wordDict)
		# self.sm.get_att(['the', 'service', 'was', 'really', 'good', 'too'])
		# self.sm.mark(['i', 'had', 'the', 'baja', 'burro', '...', 'it', 'was', 'heaven'])
		pass

	def readData(self,datafile, emotion_label_file):
		question = []
		response = []
		labels_q = [] 
		labels_r = []
		# proc .0 file (negative)
		with open(datafile, 'r') as f:
			lines = f.readlines()
		
		with open(emotion_label_file, 'r') as f:
			lines_labels = f.readlines()

		for i in range(len(lines)):
			sentences = lines[i].split('__eou__')[:-1] # there's one empty sentence in the end
			labels = lines_labels[i].split(' ')
		
			for j in range(len(sentences)-1):
				question.append(sentences[j].strip())
				response.append(sentences[j+1].strip())
				labels_q.append(labels[j])
				labels_r.append(labels[j+1])
		return question, response, labels_q, labels_r

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		question, response, labels_q, labels_r = self.data[idx]
		question_idx = self.word2index(question)
		response_idx = self.word2index(response)
		return (question_idx,response_idx,labels_q,labels_r)


	def word2index(self, sList, sos=False):
		resList = []
		for sentence in sList:
			indArr = []
			if sos:
				indArr.append(self.sos_id)
			for i in range(len(sentence)):
				word = sentence[i]
				if word in self.wordDict:
					indArr.append(self.wordDict[word])
				else:
					indArr.append(self.unk_id)
			indArr.append(self.eos_id) 
			indArr = np.array(indArr)
			resList.append(indArr)
		return resList
		


class LoaderHandler(object):
	"""docstring for LoaderHandler"""
	def __init__(self, config):
		super(LoaderHandler, self).__init__()
		print('loader handler...')	
		mode = config['opt'].mode
		config = config['loader']
		if mode == 'test':
			testData = CustomDataset(config,config['testFile'],config['testLabel'],forceNoNoise=True)
			self.ldTestEval = DataLoader(testData,batch_size=1, shuffle=False, collate_fn=seq_collate)
			return
		if mode == 'train':
			trainData = CustomDataset(config,config['trainFile'],config['trainLabel'])
			self.ldTrain = DataLoader(trainData,batch_size=config['batchSize'], shuffle=True, num_workers=2, collate_fn=seq_collate)
		# elif mode == 'val':
		devData = CustomDataset(config,config['devFile'],config['devLabel'],forceNoNoise=True)
		self.ldDev = DataLoader(devData,batch_size=config['batchSize'], shuffle=False, num_workers=2, collate_fn=seq_collate)
		self.ldDevEval = DataLoader(devData,batch_size=1, shuffle=False, collate_fn=seq_collate)
		# else:
		
