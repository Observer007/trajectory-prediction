import sys
import importlib
# importlib.reload(sys)

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rc('font', family='Times New Roman', weight=100)

# ================================== data preparing ============================================================
x = [1,2,3,4,5]
# x = [32,64,128,256,512]
name_list = ['32', '64', '128', '256', '512']

# HA_data = np.load('D:/results/HA/step-wise-rmse.npy')
# ARMA_data = np.load('D:/results/ARMA/step-wise-rmse.npy')
# VAR_data = np.load('D:/results/VAR/step-wise-rmse.npy')
# ResNet_data = np.load('D:/results/ResNet/step-wise-rmse.npy')
# MultiAttConvLSTM_data = np.load('D:/results/MultiAttConvLSTM/step-wise-rmse.npy')
# y = [HA_data,ARMA_data,VAR_data,ResNet_data,MultiAttConvLSTM_data]

dataset = 'taxi'
file_name = ['RMF', 'R2-D2', 'LSTM', 'RA-LSTM', 'MultiAttConvLSTM']
y = np.zeros((4, 5))
# for i in range(5):
#     name = dataset + '/' + file_name[i] + '/step_wise_rmse.npy'
#     y[i] = np.load(name)
'''shanghai'''
y[3] = [28.1515208576, 44.17, 64.254, 89.829, 117.66]
y[2] = [34.44378478, 51.769, 74.194, 95.603, 126.084]
y[1] = [116.71025977086003, 132.67115336204077, 144.7861431864225, 160.94802661666571, 176.15222240901625]
y[0] = [76.155138,     143.2786179,    231.25666447,   331.12486838,   457.79380771]
'''shanghai embedding size'''
y = [[76.40904, 73.41042, 68.81, 71.282, 72.376]]
'''chengdu'''
# y[0] = [43.61874717,   68.36418241,   97.56028121,  128.41090156, 163.17121463]
# y[1] = [55.30247306, 64.6381737, 72.19838774, 76.7369201, 82.75042684]
# y[2] = [43.8639311674, 51.3939, 60.1449, 69.86, 80.4127]
# y[3] = [25.9319600268, 33.7371, 42.5015, 53.1572, 64.3129]
'''chengdu'''
# y = [[46.46112,45.498,43.93,44.439,44.44]]
# ================================== settings =================================================================
figureName = './figure/shanghai_baselines.eps'

#-------------------------------------- style -----------------------------------------------------------------
#print(plt.style.available)     # print all options
plt.style.use('seaborn-paper')        # set your style

# style option
#['bmh', 'seaborn-poster', 'seaborn-talk', 'seaborn-dark-palette', 'seaborn-muted', 'seaborn-colorblind',
# 'classic', 'seaborn-white', 'seaborn-pastel', 'dark_background', 'seaborn-ticks', 'seaborn-whitegrid',
# 'seaborn-darkgrid', 'fivethirtyeight', 'seaborn-notebook', 'seaborn-dark', 'grayscale', 'ggplot',
# 'seaborn-deep', 'seaborn-paper', 'seaborn-bright']

#----------------------------------- font size ----------------------------------------------------------------
FONT_SIZE_LEGEND = 10
FONT_SIZE_AXIS = 15
FONT_SIZE_LABEL = 26
FONT_SIZE_TITLE = 26

#----------------------------------- axies setting ----------------------------------------------------------------
NUM_LINES = 1

X_LIM = [0.8, 5.2]
Y_LIM = [65, 80]

LINE_WIDTH_AXIS = 1.5

TITLE = 'Title'
LABEL_X = 'Embedding size'
LABEL_Y = 'Distance error(m)'
LABEL_TITLE = ''
FLAG_GRID = False

#----------------------------------- line setting ----------------------------------------------------------------
LINE_WIDTH = 2
MAKRER_SIZE = 7

# default color is black
LINE_COLOR = []
for i in range(NUM_LINES):
    LINE_COLOR.append(np.array([0,0,0])/255.0)

# LINE_COLOR[0] = np.array([86, 24, 27])/255
# LINE_COLOR[1] = np.array([214, 86, 42])/255
# LINE_COLOR[2] = np.array([60, 60, 60])/255
# LINE_COLOR[3] = np.array([90, 90, 90])/255
LINE_COLOR[0] = np.array([180, 0, 45])/255.0
# LINE_COLOR[1] = np.array([120, 180, 20])/255.0
# LINE_COLOR[2] = np.array([255, 127, 14])/255.0
# LINE_COLOR[3] = np.array([31, 119, 180])/255.0


# LINE_COLOR[0] = np.array([210, 210, 0])/255.0
# LINE_COLOR[1] = np.array([102, 0, 0])/255.0
# LINE_COLOR[2] = np.array([202, 98, 4])/255.0
# LINE_COLOR[3] = np.array([232, 191, 191])/255.0
# print(LINE_COLOR[0])
# LINE_COLOR[4] = np.array([1, 46, 137])/255
# LINE_COLOR[5] = np.array([103, 12, 203])/255
# LINE_COLOR[6] = np.array([213, 71, 115])/255

# default style is '-'
# '-', '--', '-.', ':', 'steps'
LINE_STYLE = []
for i in range(NUM_LINES):
    LINE_STYLE.append('-')
LINE_STYLE = ['-','-','-','-','-']

# default marker shape is '^'
MARKER_FACE_FLAG = False
MARKER_SHAPE = []
for i in range(NUM_LINES):
    MARKER_SHAPE.append('^')
MARKER_SHAPE[0] = 's';
# MARKER_SHAPE[1] = 'o';
# MARKER_SHAPE[2] = 'x';
# MARKER_SHAPE[3] = '^';
# MARKER_SHAPE[4] = 'x';
#MARKER_SHAPE[5] = '*';
# MARKER_SHAPE[6] = 'p';

# . Point marker
# , Pixel marker
# o Circle marker
# v, ^, <, > Triangle markers
# 1, 2, 3, 4, Tripod marker
# s Square marker
# p Pentagon marker
# * Star marker
# h, H Hexagon marker
# D, d Diamond marker
# | Vertical line
# _ Horizontal line
# + Plus marker
# x Cross marker

# by default, all marker faces have a color of 'w'
MARKER_FACE_COLOR = []
for i in range(NUM_LINES):
    MARKER_FACE_COLOR.append(LINE_COLOR[i])


#----------------------------------- legend setting ----------------------------------------------------------------
# legend position
# 'best' : 0, (only implemented for axes legends)
# 'upper right' : 1,
# 'upper left' : 2,
# 'lower left' : 3,
# 'lower right' : 4,
# 'right' : 5,
# 'center left' : 6,
# 'center right' : 7,
# 'lower center' : 8,
# 'upper center' : 9,
# 'center' : 10,
LEGEND_POSITION = 2

LEGEND_BOX_FLAG = True

#by default, all legend texts are set to 'linei'
LEGEND_TEXT = ['RMF','R2-D2','LSTM','RA-LSTM','MultiAttConvLSTM']
# LEGEND_TEXT[0] = 'line1'
# LEGEND_TEXT[1] = 'line2'
# LEGEND_TEXT[2] = 'line3'

# ================================== plot figure ============================================================
fig = plt.figure()
ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])
#plt.axes([0.15, 0.15, 0.75, 0.75])

lines = []
for i in range(NUM_LINES):
    line, = plt.plot(x, y[i], linestyle = LINE_STYLE[i], linewidth = LINE_WIDTH, color = LINE_COLOR[i],
                    marker = MARKER_SHAPE[i], markeredgecolor = MARKER_FACE_COLOR[i].tolist(),
                    markeredgewidth=1, markerfacecolor=MARKER_FACE_COLOR[i].tolist(),
                    markersize= MAKRER_SIZE) # alpha is the transparency

# plt.yscale('linear') # linear, log, logit, or symlog
# plt.xscale('linear') # linear, log, logit, or symlog

# Adjust the subplot layout, because the logit one may take more space
# than usual
# plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
# wspace=0.35)

# add one text to the plot
#plt.text(60, .025, r'$\mu=100,\ \sigma=15$')

# add one annotate to the plot
# plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
#              arrowprops=dict(facecolor='black', shrink=0.05),
#              )

# legend
'''matplotlib.legend.Legend(parent, handles, labels, loc=None, numpoints=None, markerscale=None,
markerfirst=True, scatterpoints=None,
scatteryoffsets=None, prop=None, fontsize=None, borderpad=None,
labelspacing=None, handlelength=None,
handleheight=None, handletextpad=None, borderaxespad=None,
columnspacing=None, ncol=1, mode=None, fancybox=None,
shadow=None, title=None, framealpha=None,
edgecolor=None, facecolor=None, bbox_to_anchor=None,
bbox_transform=None, frameon=None, handler_map=None)'''
#plt.legend(lines,LEGEND_TEXT, loc =LEGEND_POSITION, numpoints=1, markerscale=1, fontsize=FONT_SIZE_LEGEND, ncol =2, bbox_to_anchor=[0.01,0.99] )

lg = plt.legend(labels = LEGEND_TEXT, loc =LEGEND_POSITION, numpoints=1, markerscale=1, fontsize=FONT_SIZE_LEGEND, ncol =1, bbox_to_anchor=[0.01,0.99] )
if not LEGEND_BOX_FLAG:
    lg.draw_frame(False)

# matplotlib.text.Text instances
# for t in leg.get_texts():
#     t.set_fontsize(FONT_SIZE_LEGEND)  # the legend text fontsize
#     print(t.get_fontproperties())
#     fp = t.get_fontproperties()
#     fp = mpl.font_manager.FontProperties(family='times new roman', style='normal', weight=100, size=FONT_SIZE_LEGEND)
#     t.set_font_properties(fp)
#
# # set line width of legend. matplotlib.lines.Line2D instances
# for l in leg.get_lines():
#     l.set_linewidth(1.5)  # the legend line width

# Title and label
if LABEL_TITLE != None:
    plt.title(LABEL_TITLE, fontsize=FONT_SIZE_TITLE, fontname='Times New Roman' )

if LABEL_X != None:
    plt.xlabel(LABEL_X, fontsize=FONT_SIZE_LABEL, fontname='Times New Roman')
if LABEL_Y != None:
    plt.ylabel(LABEL_Y, fontsize=FONT_SIZE_LABEL, fontname='Times New Roman')

# grid
plt.grid(FLAG_GRID)

# axises
if X_LIM != None:
    plt.xlim(X_LIM)
if Y_LIM != None:
    plt.ylim(Y_LIM)
# major ticks
plt.xticks(x, name_list)
# plt.yticks([0.0, 0.2,0.4,0.6],
# 			['-5',  '-3',  '-1',  '1'])

# minor ticks
#ax = plt.gca()
ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(4))
ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(4))
plt.tick_params(which='minor', length=2, color='B')

# for the minor ticks, use no labels; default NullFormatter
#ax.xaxis.set_minor_locator(minorLocator)

for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(FONT_SIZE_AXIS)
    tick.label1.set_fontname('Times New Roman')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(FONT_SIZE_AXIS)
    tick.label1.set_fontname('Times New Roman')

for label in ax.xaxis.get_ticklabels():
    label.set_color('black')
    label.set_rotation(45)
    label.set_fontsize(FONT_SIZE_AXIS)
    label.set_fontname('Times New Roman')
for label in ax.yaxis.get_ticklabels():
    #label.set_color('red')
    #label.set_rotation(45)
    label.set_fontsize(FONT_SIZE_AXIS)
    label.set_fontname('Times New Roman')

ax.spines['bottom'].set_linewidth(LINE_WIDTH_AXIS)
ax.spines['left'].set_linewidth(LINE_WIDTH_AXIS)
ax.spines['top'].set_linewidth(LINE_WIDTH_AXIS)
ax.spines['right'].set_linewidth(LINE_WIDTH_AXIS)
for line in ax.xaxis.get_ticklines():
    # line is a Line2D instance
    #line.set_color('green')
    line.set_markersize(10)  # line length
    line.set_markeredgewidth(1.2) # line width
for line in ax.yaxis.get_ticklines():
    # line is a Line2D instance
    #line.set_color('green')     # line color
    line.set_markersize(10)      # line length
    line.set_markeredgewidth(1.2) # line width

#tight_layout automatically adjusts subplot params so that the subplot(s) fits in to the figure area
#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

foo_fig = plt.gcf()
foo_fig.savefig(figureName)
plt.show()
