#!/usr/bin/env python

def greet(hour): 
  #import pdb
  #pdb.set_trace()
  
  if hour < 12:
    print 'Good morning!'
  elif hour >= 12 and hour < 20:
    print 'Good afternoon!' 
  else:
    import pdb; pdb.set_trace() 
    print 'Good evening!'
