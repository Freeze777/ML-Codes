import pickle

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    return

def load_object(filename):
    with open(filename, 'rb') as input:
        obj=pickle.load(input)
    return obj


def balance_data(df,col_name,label1,label2):
	df_label1=df.loc[df[col_name]==label1]
	df_label2=df.loc[df[col_name]==label2]
	percentage = len(df_label1)/float(len(df_label2))
	if(percentage<=1.0):
		df_label2=df_label2.sample(frac=percentage)
		df=df_label1.append(df_label2)
	else:
		df_label1=df_label1.sample(frac=(1.0/percentage))
		df=df_label2.append(df_label1)
	df=df.reindex(np.random.permutation(df.index))
	return df

