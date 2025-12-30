text = {

"**HEATMAPS :** TorchCam" : """
#### Heatmap : TorchCam 

To generate the heatmaps seen in this application, the TorchCam library was used
in conjunction with PyTorch. The heatmap generated shows how important each 
individual picture is to help classify the object, blue pixels are irrelevant and the 
the more red the pixels are the more important they become. 

Note however that TorchCam utilize ReLU functions when generating the heat
map so pixels that have a negative impact on the classifications are set to zero 
which is blue, so the heat map doesn't tell us if a pixel goes against the classified object.   

However in actuality its not the pixel them by them selves that are important
but the effect the kernel has when applied at the pixels positions. Because the 
kernel is a small matrix all the surrounding pixels are arguably as important as 
the center pixel it self. In truth its the transformation the kernel applies on this 
group of pixels that aids the model in classifying the object correctly. 
""",


"Block 1-3" : """
#### Heatmap : Block 1-3

*To view the images select all class + heatmap black-backdrop and photo-backdrop*

###### Block 1
The first block produces the largest feature map that are the same sizes as the image,  224x224. At this stage the model appears to, in most cases, look for the general shape and contour of the object. The heatmap seem to trace the outlines of simpler shapes like the baseball, butterfly and flower with decent accuracy. The heatmap outline of dog is arguably harder to make out perhaps because the dog is more complex object or maybe it's because from this particular angle the dog lacks the strong silhuett in contrast to the butterfly and the baseball.  

This higher resolution should allow the model to pick upp the finer details in the images and when the heatmap is layered a top the photo of the cat there are some green spots following the cats whiskers, a features that could help the model to distinguish a cat from a similar shaped animals that lacks whiskers like a dog.

###### Block 2
The resolution of the feature map decreases to 112x112. The model seem to continue with edge detection but also perhaps starts to take more interest in the attributes of the objects. We can see that it pin points the individual stitches in the baseball (photo-backdrop) and traces some of the petals on the flower (black-backdrop).

###### Block 3
The feature maps are now at a 56x56 resolution. Features like eyes and noses are targeted in the two mammals. Perhaps the model can use this to get a sens of facial proportions of the model as the two eyes and nose could allow for the model to produce a triangulating effect. 

The eye patterns on the wing of the butterfly might server a similar purpose as we can see three red spots correlate with the three of the four eyes on its wings. Although the eyes might be of interest to the model simply because, like the red stitches of the baseball, this pattern is unique to the peacock butterfly which happens to be this particular species of butterfly.
""",

"Block 4-5" : """
#### Heatmap : Block 4-5

*To view the images select all class + heatmap black-backdrop and photo-backdrop*
###### Block 4
Resolution down to 28x28, the models main focus now falls on attributes such as the eyes, mouth and ears. Hard to say exactly what about these features the model finds of interest but it does look a bit like it's triangulating, we see the right eye the left part of the mouth/snout and the right part of the mouth/snout. Perhaps the model is trying to get the spacing of the cats facial features, or perhaps these edges where the cats face goes from snout to cheek or eye to fur and ear to head is useful in som way

###### Block 5
14x14 is all the resolution we get in this block. It's interesting that in all pictures, except perhaps the baseball, the heat map has been inverted, most of the points that where on interest in Block 4 is ignored in Block 5 and the areas around have become of great interest to the VGG16 model. 

This is perhaps most clear in the case of the butterfly and the flower, the center of the daisy, that has been the focal point of most blocks are now completely ignored and the shape of the entire flower has lit up. 

The butterfly is the same, they eyes look almost cut out and parts that where black in the previous block are now bright red. Perhaps the model has mined the red spots in previous blocks for all the useful information and now it focus on something new, or perhaps the lower resolution allows for detection of completely new attributes.

In the case of the cat we can see three focal points, the spot between the eyes the left cheek and the right side of the head where the ear meats the head. Also a greenish cloud now surrounds the cats head. Perhaps the shape of the head it self is of interest and its position in world around it.

Even though the dogs head is tilted away from the camera and thus is in a position different from the cat, the areas of interest are the same as in the case of the cat, the nose, the cheek and where the ear meets the head. Although the focal point of the nose is much lower and the mouth is also of great interest.

###### Combination
To make the classifications the model picks out the strongest and most useful parts of all the blocks and we can see that these combined aggregators seems to represents the blocks pretty well the most important focal points for each image is represented the strongest.

- Dog: eye, mouth and cheek. 
- Cat: eye, nose, whiskers 
- Ball: Stitches 
- Butterfly: The eye pattern
- Flower: The top petals and the yellow buds in the center
""",

"Baseball" :  """
#### Heatmap : Baseball
*To view the images referenced in this text select: class-baseball + heatmap-black-backdrop*

When we look at the black-backdrop heatmap image of the baseball we can see that, unlike with the other images, the focal point doesn't change from block to block.  The focal point  of the baseball is always constant and always on the red stitching. 

There could be several reasons for this:
1. The VGG16 model is trained to classify several different types of balls, the red stitching is a unique feature of this particular type of ball, namely a baseball, so the red stitching should allow the model do discriminate baseball from other round ball objects. 
2. A base ball is a relative simple object un like say a cat that has several different attributes like eyes, ears, paws, fur a tail etc. The baseball, on the other hand, really only have three primary attributes, its round, its whit and it has red stitching. So the number of things to focus attention on is limited and from observing the hear map we can clearly see that the model does light up the stitching and the round contours of the ball. Note however that all thought the baseball object it self is simple the context elements, of which there are none in this particular picture, might not be. These context elements could be things like baseball gloves or baseball bats and could help the model to classify the main object, aka the baseball correctly.
3. Point three is more speculative but another reason for why the model might be interested in the stitching could be that it finds it easier to determine the shape aka the roundness of the object by focusing on the stitching rather then on a white part of the ball. There are two specific parts of the stitching that seems to be of particular interest to the model, perhaps they help the model determine the shape of the object. 
""",
"**ACTIVATION MAXIMIAZTION** : Implementation" : """
#### Activation Maximization : Implementation
A small binary CNN was created and tested but the results where poor; grainy, pixely and with little to no discernable patern or shapes in the feature map that it produced. So this path was abandoned if favor of well established pre trained model.

To successfully create Activation Maximization a pre trained Convolutional Network Model was chosen namely VGG16 mainly because it's well established with a relatively basic layer implementation as basically all the layers are more or less the same, the biggest change that occurs is that the number of feature map grows while there resolution of these maps decreases. 

To create the feature maps the last layer was removed, which is the layer that turns the feature maps in to classifications. The layers where then grouped in to 5 blocks this lined up nicely with the pooling layers that lowers the resolution and increased the number of feature maps produced. 

Gradient Ascent was then implemented in an effort to transform the input image to maximize lose, this way finished image becomes abstract representation that maximizes the activation of the feature maps for that particular class.

A randomized pixel image is generated and along with a class is feed in to the CNN, for each block the noise image is transformed to maximize these feature maps of that particular block,  three feature maps that activates the most from each block is selected and the image is passed to the next block.

This process produce one feature map for every kernel in the VGG16 model, so 4,224 in total. To select the most important feature map a *Forward Hook* is implemented that selects the feature maps and then employs a *Global Average Pooling* to reduces each map down to a score, the more the feature map is *activated* by the class the better the score. The three feature maps with the highest score is picked from each block and then from these three feature maps one is arbitrarily selected based on how interesting or unique the analyst finds it to be.


To produce the more vivid and high resolution version of the feature map the script utilizes two distinct techniques *Multi-Scale (Octave) Optimization* and *Image Restoration*
Because the feature maps in the later blocks have a very low resolution, some as low as 14x14, we can't simply just scale them up, this would only result in large blurry images so instead we use *Multi-Scale Optimization* which essentially increases the size of the feature map, in several smaller steps, after it scales the image up one step it optimizes the image and continues this process until the image reaches the desired resolution. The *Laplacian Sharpening* sharpens things like edges each time the image gets upscaled. A jitter is also employed that moves the image a few pixel in a random direction this helps reduce artifacts.
To make the feature map look more vivid and less gray and noisy we use TV regularization that punishes random jagged pixels and thus produces a more coherent feature map so in a sens, while *Laplacian* sharpens the image, *TV* smooths it out.
""",

"Feature Maps" : """
#### Activation Maximization : Feature Maps
The feature maps are produced as the kernel, a small matrix, is drawn across the image. For each block the number of these maps increase. In the three first blocks the same feature maps appears over and over again regardless of the object in the image. This is most notice in block 2 where in the four different types of feature maps where among the top 3 fore each of these pictures, although this is a tiny test sample of five objects it does strengthen the case that these top high resolution layers are more interested in basic feature that they all share like edges, shapes, textures unlike that later blocks where we see a greater verity of feature maps, which might indicate that these look at attributes more unique to that particular class.

We can also see that the pattern in the feature maps becomes more complex the further down the block we go, the first block is basically just lines in different direction.

Then in block 2 we get simple geometric patterns like checker board, mace pattern and stripes, while block 3 and 4 has more texture to them, there still patterns all though in block 4 it seems to transition to parts and in block 5 it looks less like a pattern and more like a collage of different parts.
""",

"Noise Input" : """
#### Activation Maximization : Noise
*To view the images select all class + maximization noise*

All tough the final maximized image is exceedingly abstract, the transference from block 5 to the finally image looks to be there, the most clear example of this is probably in the butterfly image, we see the checkered pattern roundish highlighted parts in the left corner in both the feature map of block 5 and the final image.  

It's more difficult to pick out any specific attribute of the target class. If we look at the maximized baseball we can discern a roundish shape and the stitch pattern. However if we compare this image to the maximized image of the dog, we see a uncanny resembles, which puts in question both the rounded shape and the stitching. In truth one could argue that to the human eye the maximized image of a dog and a baseball is much more similar to each other then to the classes the represents. So it's exceedingly easy to fool once self in to believing that we see patterns that re not really there.

If we look at the butterfly image however it does seem like there are som wing pattern emerging, although again this could be illusional trickery, our mid finding paterns where there are none. 

The flower is perhaps our best candidate for actually representing the class object, we can see wat looks like flower buds in the middle,  a roundish flower shape with petals emerging.

To gain a more clear insight, one could generate hundreds of image of each class and there on might start to recognize attributes of the object with more certainty.
""",

"Image Input" : """
#### Activation Maximization : Image
*To view the images select all class + maximization photo*

We don't need to pass in a noise image, any image we put in will be morphed in order to maximize for the input class. In this example we put in the image of the image class.

In the first block, we don't see much different from the noise, the images has been completely reduced down to lines. 

But in block two we can see faint remanents of the original images, perhaps the most interesting is the butterfly where the mace like pattern looks like it takes on the shape of the butterfly, but this so so vague that it could be a case of the observer seeing a pattern where there is non. 

In block three the effect is more clear, the pattern has morphed with the butter fly and the leather like pattern clearly outlines the base ball and follows the  stiches of the ball.

In block four the pattern seems to start to contain attributes of the object it self, the baseball have a stitch like feature map. The flour have bubbles reminiscent of the bud like center part. The cat lights up  something that looks a bit like fru and the dog has something that is not unlike eyes.

In block five the distortion is a lot milder as the images still looks a lot like the original, the ball, butterfly and flower looks like an artistic rendering of the original image. The cat also appear clearly for the first time it's no longer simply a brighter version of the pattern as in the previous blocks. 

The final block is similar to the previous block five but if we  look at the dog it looks like it contain multiple dog objects we can see the snout appear in at least two different angles. It could be argued, at least in these cases that the combine blocks looks more deformed than block five. Because the VGG16 model is trained to classify subclasses so not just cat, dog, flower but specific cats, dogs and flower, in this case Egyptian, Alsatian, Daisy then perhaps block five tries to classify the general object type and thus generating a more cohesive image while the combined block focuses more on the specific class and thus becomes more abstract. Or perhaps the combined block becomes more abstract and deformed simply because it tries to incorporate a multitude of feature maps.
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
