# EEG-EMG-based-Brain-Computer-Interface with Generative Adversarial Networks (GANs)


## Overview

This project focuses on improving Brain-Computer Interface (BCI) systems using Generative Adversarial Networks (GANs). BCIs enable users to interact with devices using their brain activity, and one significant challenge in BCI development is the long calibration time required for the system to adapt to the user's brain signals. This project explores the use of GANs to generate synthetic EEG (Electroencephalographic) and EMG (Electromyographic) data for augmenting the original dataset, aiming to reduce calibration time and improve classification accuracy.

## Motivation

Recent technological advancements have increased interest in BCI systems across various fields, including rehabilitation, gaming, and education. However, long calibration times hinder widespread adoption of these systems. This project aims to address this challenge by leveraging GANs to generate synthetic EEG and EMG data, thereby reducing the reliance on collecting extensive real-world data from users.

## Methodology

Three types of GANs were investigated in this project:

1. Standard GAN
2. Deep Convolutional GAN (DCGAN)
3. Wasserstein GAN (WGAN)

These GANs were trained to generate synthetic EEG and EMG data that mimic the patterns observed in real brain and muscle signals during Motor Imagery tasks and EM-GAN focusing on generating synthetic electromyography (EMG) signals.
