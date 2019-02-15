import numpy as np
import torch
import os

def subset(array):
    result = []
    n = len(array)
    for k in range(1, n):
        for i in range(n-k+1):
            result.append(array[i:i+k])
    return result

def seq_collate(batch):
	# print('>>>>>>>batch: '+str(batch))
	batchSize = len(batch)
	def extract(ind):
		maxLen = 0
		lengths = []
		for seq in batch:
			seqLen = len(seq[ind])
			lengths.append(seqLen)
			if seqLen > maxLen:
				maxLen = seqLen
		packed = np.zeros([batchSize, maxLen])
		for i in range(batchSize):
			packed[i][:lengths[i]] = batch[i][ind]
		lengths = np.array(lengths)
		# inds = np.argsort(lengths)[::-1]
		return torch.LongTensor(packed), torch.tensor(lengths)

	def extract_marker_lengths(ind):
		lengths = []
		maxlen = 0
		for seq in batch:
			numMk = len(seq[ind])
			maxlen = max([maxlen,numMk])
		lengths = np.zeros([batchSize,maxlen])
		k = 0
		for seq in batch:
			numMk = len(seq[ind])
			tmp = [len(seq[ind][i]) for i in range(numMk)]
			lengths[k,:numMk] = np.array(tmp)
			k += 1
		return torch.tensor(lengths)

	question, qLengths = extract(0)
	response, rLengths = extract(1) 
	labels_q, lqLengths = extract(2)
	labels_r, lrLengths = extract(3)

	return {'question': question,
			'qLengths': qLengths,
			'response': response,
			'rLengths': rLengths,
			'labels_q': labels_q,
			'lqLengths':lqLengths,
			'labels_r': labels_r,
			'lrLengths':lrLengths
 	}

def reloadModel(model,config):
	checkpoint = os.path.join(config['contPath'], config['opt'].resume_file)
	print("=> Reloading checkpoint '{}': model".format(checkpoint))
	checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
	# model.load_state_dict(self.checkpoint['state_dict'])
	model_dict = model.state_dict()
	# 1. filter out unnecessary keys
	pretrained_dict = {}
	for k, v in checkpoint['state_dict'].items():
		if(k in model_dict):
			pretrained_dict[k] = v
	# 2. overwrite entries in the existing state dict
	model_dict.update(pretrained_dict)
	# 3. load the new state dict
	model.load_state_dict(model_dict)
	return model

def makeInp(inputs):
	if torch.cuda.is_available():
		for key in inputs:
			inputs[key] = inputs[key].cuda()
	return inputs
