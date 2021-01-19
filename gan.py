import streamlit as st
import pandas as pd 
import numpy as np
import time
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt 
import os
from discriminater import make_discriminator_model
from generator import make_generator_model
checkpoint_dir = './checkpoints'
inputshape=10

generator = make_generator_model(inputshape)
discriminator = make_discriminator_model()

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,discriminator_optimizer=discriminator_optimizer,generator=generator,discriminator=discriminator)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

f1=st.sidebar.slider(" ", -1.0, 1.0, step=0.01,key="sl1")
f2=st.sidebar.slider(" ", -1.0, 1.0, step=0.01,key="sl2")
f3=st.sidebar.slider(" ", -1.0, 1.0, step=0.01,key="sl3")
f4=st.sidebar.slider(" ", -1.0, 1.0, step=0.01,key="sl4")
f5=st.sidebar.slider(" ", -1.0, 1.0, step=0.01,key="sl5")
f6=st.sidebar.slider(" ", -1.0, 1.0, step=0.01,key="sl6")
f7=st.sidebar.slider(" ", -1.0, 1.0, step=0.01,key="sl7")
f8=st.sidebar.slider(" ", -1.0, 1.0, step=0.01,key="sl8")
f9=st.sidebar.slider(" ", -1.0, 1.0, step=0.01,key="sl9")
f10=st.sidebar.slider(" ", -1.0, 1.0, step=0.01,key="sl10")

#out_images = np.array([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10])
inp=tf.constant(np.array([[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10]]).astype('float32'))
pred=generator(inp)
im=pred.numpy()
#im
st.image(im[0,:,:,0:3], caption='Face',clamp=True,use_column_width=True)
#pred