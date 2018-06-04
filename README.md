# Optical Character Recognition Microservice
Optical Character Recognition microservice for MARAGI

## Input

Input is a numpy array with the following characteristics:
* 32x32 size
* grayscale (0-255) 
* white = 255, black = 0

## Output

List of predicted characters where each:
* Certainty (0..1) where 1 is 100%
* Character

