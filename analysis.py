text = {

"**HEATMAPS :** TorchCam" : """
#### Heatmap : TorchCam 

To generate the heatmaps seen in this application, the TorchCam library was used in conjunction with PyTorch. The generated heatmaps illustrate the importance of each individual pixel; red pixels have a greater impact on the classification of the object while bluer pixels have less impact. 

Note that TorchCam utilizes ReLU functions when generating the heatmaps so pixels with negative values are set to zero. For example if the classified object is a baseball but a pixel thinks it to be a tennis ball and thus would have a negative value in context of the classification of the baseball that pixel will still be set to blue.  

However, in actuality, its not the pixels themselves that are important for classification but the effect the kernel has when applied at the pixel's positions. Because a kernel is a small matrix that covers multiple pixels then these surrounding pixel are as essential as the center pixel itself. In truth its the transformation the kernel applies on this group of pixels; this process *convolves* the input image to produce a *feature map*. This *feature map* is a representation of the activation of a neuron which aids the model in classifying the object correctly so it's not the RGB value of each individual pixel that matters. One could say that it's the context of the pixel that is important rather than the pixel itself. 
""",


"Block 1-3" : """
#### Heatmap : Block 1-3

*To view the images select all class + heatmap black-backdrop and photo-backdrop*

To try and get some insight in what the *feature maps* do, we can look at the *heatmaps* generated from each block.
###### Block 1
The first block produces the largest feature map that are the same size as the image (**224x224**), meaning it produces the maps with the *highest spatial fidelity*. At this stage the model appears to, in most cases, look for the general shape and contour of the object. The heatmap seem to trace the outlines of simpler shapes like the baseball, butterfly and flower with decent accuracy. The heatmap outline of dog is arguably harder to make out perhaps because the dog is more complex object or maybe it's because from this particular angle the dog lacks the strong silhouette in contrast to the butterfly and the baseball.  

This higher resolution should allow the model to pick up the finer details in the images and when the heatmap is layered atop the photo of the cat there are some green spots following the cats whiskers, a feature that could help the model to distinguish a cat from a similar shaped animals that lacks whiskers like a dog.

###### Block 2
The resolution of the feature map decreases to **112x112**. The model seem to continue with edge detection but also perhaps starts to take more interest in the attributes of the objects. We can see that it pin points the individual stitches in the baseball (photo-backdrop) and traces some of the petals on the flower (black-backdrop).

###### Block 3
The feature maps are now at a **56x56** resolution. Attributes like eyes and noses are targeted in the two mammals. Perhaps the model can use this to get a sense of facial proportions of the model as the two eyes and nose could allow for the model to produce a *triangulating effect*. 

This block picks up on *discriminative features* like the eye patterns on the wing of the butterfly which might serve a similar purpose as we can see three red spots correlate with the three of the four "eyes" on its wings. Although the "eyes" might be of interest to the model simply because, like the red stitches of the baseball, this pattern is unique to the peacock butterfly which happens to be this particular species of butterfly.
""",

"Block 4-5" : """
#### Heatmap : Block 4-5

*To view the images select all class + heatmap black-backdrop and photo-backdrop*
###### Block 4
Resolution is down to **28x28** pixels, the model's main focus now falls on attributes such as the eyes, mouth and ears. While it's hard to say exactly what about these features the model finds of interest but it does look a bit like it's triangulating, we see the right eye the left part of the mouth/snout and the right part of the mouth/snout. Perhaps the model is trying to get the spacing of the cat's facial features, or perhaps these edges where the cats face goes from snout to cheek or eye to fur and ear to head is useful in some way.

###### Block 5
**14x14** is all the resolution we get in this block. It's interesting that in all pictures, except perhaps the baseball, the heat map has been "inverted", most of the points that were of interest in Block 4 is ignored in Block 5 and the areas around have become of great interest to the VGG16 model. 

This is perhaps most clear in the case of the butterfly and the flower, the center of the daisy, that has been the focal point of most blocks are now completely ignored and the shape of the entire flower has lit up. 

The butterfly is the same, the eye pattern on the wings look almost cut out and parts that where black in the previous block are now bright red. Perhaps the model has mined the red spots in previous blocks for all the useful information and now it focus on something new, or perhaps the lower resolution allows for detection of completely new attributes.

In the case of the cat we can see three focal points, the spot between the eyes the left cheek and the right side of the head where the ear meets the head. Also a greenish cloud now surrounds the cats head. Perhaps the shape of the head itself is of interest and its position in world around it.

Even though the dogs head is tilted away from the camera and thus is in a position different from the cat, the areas of interest remains the same, in the case of the cat; the nose, the cheek and where the ear meets the head. Although the focal point of the nose is much lower and the mouth is also of great interest.

###### Combination
To make the classifications the model picks out the strongest and most useful parts of all the blocks and we can see that these combined aggregators seems to represents the blocks pretty well the most important focal points for each image is represented the strongest.

- **Dog:** eye, mouth and cheek. 
- **Cat:** eye, nose, whiskers 
- **Baseball:** Stitches 
- **Butterfly:** The eye pattern
- **Flower:** The top petals and the yellow buds in the center
""",

"Baseball" :  """
#### Heatmap : Baseball
*To view the images referenced in this text select: class-baseball + heatmap-black-backdrop*

When we look at the black-backdrop heatmap image of the baseball we can see that, unlike with the other images, the focal point doesn't change from block to block.  The focal point  of the baseball is always constant and always on the red stitching. 

There could be several reasons for this:
1. **Discriminative Features:** The VGG16 model is trained to classify several different types of balls, the red stitching is a unique feature of this particular type of ball, namely a baseball, so the red stitching should allow the model to discriminate baseball from other round ball objects such as tennis balls which lack the red stitching. 

2. **Object Simplicity and Context:** A baseball is a relative simple object unlike say a cat that has several different attributes like; eyes, ears, paws, fur a tail etc. The baseball, on the other hand, really only have three primary attributes, its round, its white and it has red stitching. So the number of things to focus attention on is limited and from observing the heatmap it is clearly that the model does light up the stitching and the round contours of the ball. Note however that although the baseball object itself is simple the context elements, of which there are none in this particular picture, might not be. These context elements could be things like; baseball gloves, baseball bats, baseball player, and could help the model to classify the main object, aka the baseball correctly.

3. **Spatial Gradients and Shape:** Point three is more speculative but another reason for why the model might be interested in the stitching could be that it finds it easier to determine the shape aka the roundness of the object by focusing on the stitching rather than on a white part of the ball. There are two specific parts of the stitching that seem to be of particular interest to the model, perhaps they help the model determine the shape of the object. 
""",
"**ACTIVATION MAXIMIAZTION** : Implementation" : """
#### Activation Maximization : Implementation

###### Model Selection
A small binary CNN was created, trained and tested but the results were grainy poor and pixely feature maps that contained patterns so weak as to be non existent. This path was abandoned in favor of  VGG16 a well established pre trained model.

This model was powerful enough to create detailed feature map to illustrate the *activation maximization* of the *neural network*. Another benefit of utilizing the VGG16 is that the model had relatively basic layer implementation as layer aka neuron is basically all the same. The biggest change that occurs between the layers are that the number of feature map grows while the resolution of these maps decreases. 

###### Maximization Implementation
To turn the VGG16 model from a classifier and in to a *feature map generator* the last layer aka the layer responsible for making the classification based on the feature map activation was removed from the model. The layers were then grouped in to 5 blocks, this lined up nicely with the pooling layers that lowers the resolution and increased the number of feature maps produced. 

*Gradient Ascent* was then implemented in an effort to transform the input image to maximize the *activation score*. This way the finished image became an abstract representation that maximizes the activation of the *feature maps* for that particular class.

###### Image Class Maximization
To generate an image that maximized the *feature map* for a particular class one of the one thousand classes that VGG16 had been pre trained to classify was fed in to the model alongside an image. This image could theoretically be anything, even simply a completely black or white image would do, but its easier for the model if it's a picture of randomized pixel noise, this gives the model some traction. For each class represented in this application two image was feed in to the model, a noise pixel image as well as a photo of the class, this dual image approach will theoretically provide some more context on how the model generates these maximized class images. 

###### Maximized Image Selection
This resulted in one *feature map* for every kernel in the VGG16 model, so 4,224 in total for every picture that is fed to the model. In order to make the results more easy to digest *Forward Hook* is implemented that selects the *feature maps* by employing a *Global Average Pooling* to reduces each map down to a score and then picking the *feature map* with the highest score, the more the *feature map* is *activated* by the class the better the score. The three *feature maps* with the highest score is picked from each block and then from these three *feature maps* one is arbitrarily selected based on how interesting or unique the analyst found them to be.

###### Maximized Image Enhancement
To produce more vivid and high resolution version of the these images the script utilizes two distinct techniques *Multi-Scale (Octave) Optimization* and *Image Restoration*
Because the feature maps in the later blocks have a very low resolution, some as low as 14x14, we can't simply just scale them up, this would only result in large blurry images so instead we use *Multi-Scale Optimization* which essentially increases the size of the *feature map* by breaking the upscaling process down in to several smaller steps. Each step the image is only scaled up a little bit each time then the image is optimized and so it continues until the image reaches the desired resolution. 

To optimize the image *Laplacian Sharpening* was utilized to, no one's surprise, sharpen elements in the image, elements like edges. A *Jitter* is also employed, a *Jitter* moves the image a few pixel in a random direction this helps reduce artifacts.

*TV Regularization* was used to make the *feature map* look more vivid and less gray and noise by  punishing random jagged pixels and thus produces a more coherent feature map so in essence, *Laplacian* sharpens the image and *TV* smooths it out again and this back and forth process helps create a large, sharp and vivid representation of the *feature map*.
""",

"Feature Maps" : """
#### Activation Maximization : Feature Maps
*To view the images select all class + maximization noise*

The *feature maps* are produced as the kernel, a small matrix, is "raked" (convolved) across the input image. For each block the number of these maps increases. In the three first blocks the same feature maps appears over and over again regardless of the object in the image. Although this is a tiny test sample of only five objects it does attest to the strengthe these *feature maps*. 

We can see similarities in the *feature map* from each block, the high resolution layers are more interested in basic feature that they all share like edges, shapes, textures unlike that later blocks where we see a greater variety of *feature maps*, which might indicate that these look at attributes more unique to that particular class.

We can also see that the pattern in the *feature maps* becomes more complex the further down the block we go, the first block is basically just lines in different directions.

Then in block 2 we get simple geometric patterns like checkerboard, mace pattern and stripes, while block 3 and 4 has more texture to them, there still patterns although in block 4 it seems to transition to the parts that make up the object and in block 5 it looks less like a pattern and more like a collage of different elements.
""",

"Noise Input" : """
#### Activation Maximization : Noise
*To view the images select all class + maximization noise*

When a pixelized representation of noise is used as an input image the final maximized image becomes exceedingly abstract but the transference from block 5 to the final image looks to be there, the most clear example of this is probably in the butterfly image, we see the checkered pattern roundish highlighted parts in the left corner in both the feature map of block 5 and the final image.  

It's more difficult to pick out any specific attribute of the target class. If we look at the maximized baseball we can discern a roundish shape and the stitch pattern. However if we compare this image to the maximized image of the dog, we see a uncanny resemblance, which puts into question both the rounded shape and the stitching. In truth, one could argue that to the human eye, the maximized image of a dog and a baseball is much more similar to each other than to the classes they represents. So it's exceedingly easy to fool oncsself in to believing that we see patterns even when there  not really there.

If we look at the butterfly image however it does seem like there are some wing pattern emerging, although again this could be down to illusional trickery, and its our mind that is finding patterns where there are none. 

The flower is perhaps our best candidate for actually representing the class object, we can see what looks like flower buds in the middle, a roundish flower shape with petals emerging.

To gain a more clear insight, one could generate hundreds of image of each class and from there might start to recognize attributes of the object with more certainty.
""",

"Image Input" : """
#### Activation Maximization : Image
*To view the images select all class + maximization photo*

We don't need to pass in a noise image, any image we put in will be morphed in order to maximize for the input class. In this example we put in the photo of the class.

**Block 1**
In the first block, we don't see much different from the noise, the images has been completely reduced down to lines. 

**Block 2**
We can see faint remnants of the original images, perhaps the most interesting is the butterfly where the mace like pattern looks like it takes on the shape of the butterfly, but this so so vague that it could be a case of the observer seeing a pattern where there are none. 

**Block 3**
The effect is more clear, the pattern has merged with the butter fly and the leather like pattern clearly outlines the baseball and follows the stiches of the ball.

**Block 4**
The pattern seems to start to contain attributes of the object itself, the baseball has a stitch like feature map. The flower have bubbles reminiscent of the bud like center part. The cat lights up  something that looks a bit like fur and the dog has something that is not unlike eyes.

**Block 5**
The distortion is a lot milder as the images still looks a lot like the original, the ball, butterfly and flower looks like an artistic rendering of the original image. The cat also appears clearly for the first time it's no longer simply a brighter version of the pattern as in the previous blocks. 

**Combined**
The final block is similar to the previous block 5 but if we look at the dog it looks like it contain multiple dog objects we can see the snout appear in at least two different angles. It could be argued, at least in these cases that the combined blocks looks more deformed than block five. Because the VGG16 model is trained to classify subclasses so not just cat, dog, flower but specific cats, dogs and flower, in this case Egyptian, Alsatian, Daisy then perhaps block five tries to classify the general object type and thus generating a more cohesive image while the combined block focuses more on the specific class and thus becomes more abstract. Or perhaps the combined block becomes more abstract and deformed simply because it tries to incorporate a multitude of *feature maps*.
""",

"**CREDITS**" : """
#### Credits

#### AI
Large parts of the code used in this projected where generated or enhanced with the assistance of Googles AI model Gemini 3.

#### Images
Image : Flower - Daisy 
Set : [Flowers-Dataset](https://www.kaggle.com/datasets/imsparsh/flowers-dataset)
Site : [Kaggle](www.kaggle.com)
Downloaded : 2025-12-20
License : [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)
Publisher : Sparsh Gupta

Image : Ball - Baseball 
Set : [Sports-Balls](https://www.kaggle.com/datasets/samuelcortinhas/sports-balls-multiclass-image-classification)
Site : [Kaggle](www.kaggle.com)
Downloaded : 2025-12-17
License : [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)
Publisher : Samuel Cortinhas

Image : Butterfly - Peacock 
Set : [Butterfly-Image-Classification](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification)
Site : [Kaggle](www.kaggle.com)
Downloaded : 2025-12-16
License : [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)
Publisher : DePie

Image : Dog - Alsatian
Set : [Cats & Dogs](https://www.kaggle.com/datasets/ashfakyeafi/cat-dog-images-for-classification)
Site : [Kaggle](www.kaggle.com)
Downloaded : 2025-12-20
License : [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)
Publisher : Ashfak Yeafi

Image : Cat - Egyptian 
Set : [Cats & Dogs](https://www.kaggle.com/datasets/ashfakyeafi/cat-dog-images-for-classification)
Site : [Kaggle](www.kaggle.com)
Downloaded : 2025-12-20
License : [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)
Publisher : Ashfak Yeafi

""",
}
