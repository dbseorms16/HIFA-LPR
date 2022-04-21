# readline_all.py
import numpy as np
f = open("DBPN_avg_loss.txt", 'r')
num=[]
i=0
while True:
    line = f.readline()
    if(i<10): 
    	line=line[35:]
    elif(i<100):
    	line=line[36:]
    else:
    	line=line[37:] 
    
    if not line: break
    i=i+1
    new=(float)(line)
    num.append(new)
    print(num)
    


from matplotlib import pyplot as plt
plt.figure()
    

    

plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("DBPN_loss_graphx4")
Y = num
plt.plot(Y,color='red', linestyle='dashed')



plt.show()

#plt.plot(num)
#plt.plot(x)
#plt.xlabel("fxkk")
#plt.ylabel("fdfd")
#
#plt.legend(["fxxk"])
plt.show()    
f.close()