With a pretrained ALiBi model, trying:

1. Local Window Attention
In general, model is surprisingly robust to these changes (usually see a PPL increase of only 1 to 2 points)

Base: 28.57

Window Sizes:
32,32,None,None,None,None, None, None : 31.393389
128,128,None,None,None,None, None, None: 29.2364
256,256,None,None,None,None, None, None: 28.72
256,256,256,256,None,None, None, None: 29.088
256,None, None, None, 256,None, None, None: 30.6976
128, None, None, None, None, None, None, None: 29.066
256, None, None, None, None, None, None, None: 28.67
512, None, None, None, None, None, None, None: 28.59
512, None, 512, None, 512, None, 512, None: 29.42


2. Changing Distance Matrix from Linear to Sqrt
WikiText2 PPL: 28.566505 -> 267.46466