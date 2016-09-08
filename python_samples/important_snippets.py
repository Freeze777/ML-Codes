		    def allindices(string, sub):
			   l=[]
			   i = string.find(sub)
			   while i >= 0:
			      l.append(i)
			      i = string.find(sub, i + 1)
			   return l