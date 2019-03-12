import matplotlib.pyplot as plt
import numpy as np
#chengdu
'''
tmp =[20.9125682558, 54.4124542487, 101.041235809, 158.942452665, 230.893286441, 315.82212588, 413.003577314, 522.509725597]
for i in range(1, len(tmp)):
    tmp[len(tmp)-i] = tmp[len(tmp)-i]-tmp[len(tmp)-i-1]
all2 = tmp
only_e = [23.82550555603385, 37.49400888651051, 48.992515637554625, 63.34024300443589, 76.82931711466244, 88.68625643114345, 103.65624964276562, 120.9502933765747]
only_f = [28.97693801707924, 41.059445794655424, 56.190005906074795, 69.78358609154316, 80.447557219498, 95.62711889390195, 109.51659504851831, 122.88490599527196]
none = [37.66496708555437, 51.48765832393677, 66.0891933545713, 79.39708045718491, 96.20170509368742, 106.18643891860954, 117.7671614162791, 131.24841141919964]
R2D2 = [69.7937139133999,79.93998489200378,89.74432371603653,98.94308851484436,
	   104.75229868069674,107.04142977006025,110.08937724171545,117.39010035107823]
# R2D2 = [56.16886102640845,75.5250585773212,87.22574743312063,88.57547695220279,84.2292413407186]
RMF = [43.61874717,   68.36418241,   97.56028121,  128.41090156,
  163.17121463,  204.16944481,  247.233177,    291.54086603]
# cRALSTM = [21.53181424720069, 33.25423283037415, 41.826833433830494, 50.305289790342506, 54.94969303985015]
cRALSTM = [19.406917792217136, 27.911169037908955, 30.56509898317235, 32.32072544254303, 31.185231573827778]
'''
all2 = []
plt.figure(1)
x = range(1, 6)
# plt.plot(x, RMF[:5], label='RMF', marker='s')
# plt.plot(x, R2D2[:5], label='R2-D2', marker='o')
plt.plot(x, none[:5], label='LSTM', marker='x')
plt.plot(x, only_f[:5], label='IntraFea-LSTM', marker='x')
plt.plot(x, only_e[:5], label='InterFea-LSTM', marker='v')
plt.plot(x, all2[:5], label='RA-LSTM', marker='^')
# plt.plot(x, cRALSTM, label='RA-LSTM', marker='^')





# plt.plot(x, cRALSTM, color='orangered', label='RA-LSTM filter', marker='s')
plt.legend(loc='upper left',fontsize=13)
plt.xlabel("Prediction step",fontsize=15)
plt.ylabel("Distance error(m)", fontsize=15)
plt.xticks(range(1, 11), fontsize=13)
plt.yticks(fontsize=13)
plt.xlim(0.8, 5.2)
plt.ylim(10, 120)
# plt.ylim(0, 250)
print(np.sum(np.array(RMF[:5])), np.sum(np.array(R2D2[:5])), np.sum(np.array(none[:5])),
             np.sum(np.array(only_f[:5])), np.sum(np.array(only_e[:5])), np.sum(np.array(all2[:5])))
plt.show()
'''
#shanghai
all2 = [37.82935317408641, 60.76098285813929, 84.4632232850198, 113.23269771498826, 142.61405054852543]
only_e = [42.157255203142775, 61.3519505423198, 84.61444110463388, 111.4199588356521, 141.44366641882866]
only_f = [50.981523745777785, 76.82800311836593, 107.4722274049382, 138.53035053676766, 169.95866321534132]
none = [73.92881273692743, 94.58835804873095, 121.04323821618841, 147.01586255453148, 176.0979698756225]
R2D2 = [116.71025977086003,132.67115336204077,147.7861431864225,166.94802661666571,182.15222240901625]
RMF = [76.155138,     143.2786179,    231.25666447,   331.12486838,   457.79380771]
cRALSTM = [36.50772024476504, 51.10494153359608, 64.07076418224408, 65.36866829189209, 65.94091001495771]
plt.figure(1)
x = range(1, 6)
'''
# plt.plot(x, RMF[:5], label='RMF', marker='s')
# plt.plot(x, R2D2[:5], label='R2-D2', marker='o')
plt.plot(x, none[:5], label='LSTM', marker='x')
plt.plot(x, only_f[:5], label='IntraFea-LSTM', marker='x')
plt.plot(x, only_e[:5], label='InterFea-LSTM', marker='v')
plt.plot(x, all2[:5], label='RA-LSTM', marker='^')
# plt.plot(x, cRALSTM, label='RA-LSTM', marker='^')
# plt.plot(x, cRALSTM, 'orangered', label='RA-LSTM filter', marker='s')
plt.legend(loc='upper left', fontsize=13)
plt.xlabel("Prediction step", fontsize=15)
plt.ylabel("Distance error(m)", fontsize=15)
plt.xticks(range(1, 11), fontsize=13)
plt.yticks(fontsize=13)
plt.xlim(0.8, 5.2)
plt.ylim(20, 200)
# plt.ylim(0, 500)
print(np.sum(np.array(RMF)), np.sum(np.array(R2D2)), np.sum(np.array(none)),
             np.sum(np.array(only_f)), np.sum(np.array(only_e)), np.sum(np.array(all2)))
plt.show()


r2d2_shanghai = [0.925, 0.8725, 0.7225, 0.5125, 0.3675]
# r2d2_chengdu  = [0.945, 0.816, 0.581, 0.366, 0.244]
our_rate_shanghai = [0.97619051, 0.95238098, 0.89880955, 0.80059527, 0.71130955]
# our_rate_shanghai1 = [ 0.97619051, 0.94345241, 0.86904765, 0.76190479, 0.66071432]
# our_rate_chengdu = [0.96592847,  0.955707,    0.92504261,  0.89948894,  0.85178877]
our_rate_chengdu1 = [0.94718911, 0.91311756, 0.8194208, 0.7512777, 0.6678024]
plt.figure(3)
# plt.bar(np.arange(1,6)-0.1, our_rate_shanghai, width=0.2, color='b', label='RA-LSTM')
# plt.bar(np.arange(1,6)+0.1, r2d2_shanghai, width=0.2, color='y', label='R2-D2')
plt.bar(np.arange(1,6)-0.1, our_rate_shanghai, width=0.2, color='b', label='RA-LSTM')
plt.bar(np.arange(1,6)+0.1, r2d2_shanghai, width=0.2, color='y', label='R2-D2')
plt.xlim(0.4, 5.6)
plt.ylim(0, 1)
plt.legend(loc='upper right', fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('Prediction step', fontsize=15)
plt.ylabel('Prediction rate', fontsize=15)
plt.show()
