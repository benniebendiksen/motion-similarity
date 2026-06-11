# motion-similarity
A Python project for learning a perceptually-accurate latent space for motion synthesis wherein geometric distance is monotic with dissimilarity ratings between inputs. This project trains models on motion capture data as well as data gathered from user studies. It employs an adaptation of the triplet network via a custom loss over user study data; this permits us to begin learning a perceptual similarity metric over human motion.  

Final proposed network architecture is as follows: 

[![Model Architecture](./images/final_architecture.png)](./images/final_architecture.png)

See the original motivating paper [here](https://drive.google.com/file/d/1x_s68q_QcSxHmW7XdQGjGIGRS34B5cop/view).

All motion data is stored as .bvh files; read about the .bvh format [here](https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/BVH.html)

## On LMA
Part of the features inputted into our similarity network represent style information. This information is produced by our CNN and comprises the Effort category of Laban Movement Analysis (LMA). LMA is a technique created by Rudolf Laban to formally describe human movement. It is used in a broad range of fields such as dance, physical therapy, drama, psychology, and anthropology. LMA comprises four categories: Body, Effort, Shape, and Space. LMA terms are capitalized.  

**Body** defines the structural aspects of the human body during motion, i.e., what the body is doing and the relationship of the body parts during motion. 

**Effort** is the dynamic component, which is used to describe the characteristics of movement based on humans’ inner attitudes. 

**Shape** determines the way these attitudes are expressed through the body, and it is manifested in postures. 

**Space** describes how a person connects to their environment; locale, directions, and paths of a movement, and it is partly related to steering. 

PERFORM keeps Body and Space fixed and focuses on Shape and Effort components. In fact, it considers Shape regarding its relationship to Effort. So, it wouldn’t be wrong to conclude that we are only focusing on Effort as the dynamic component of LMA in our work.

### On Effort
Effort is described through four motion factors, where each factor is a continuum between bipolar Effort elements: indulging and condensing and can be represented as ranging from -1 to +1. The Effort elements are Space (Indirect vs. Direct), Weight (Light vs. Strong), Time (Sustained vs. Sudden), and Flow (Bound vs. Free).

Each effort element is characterized by certain trait-descriptive adjectives as:

Indirect: Flexible, meandering, multi-focus
Direct: Single-focus, channeled, undeviating 
Light: Buoyant, delicate 
Strong: Powerful, having an impact 
Sustained: Lingering, leisurely, indulging in time 
Sudden: Hurried, urgent 
Free: Uncontrolled, abandoned, unlimited 
Bound: Careful, controlled, restrained Human beings exhibit a variety of Effort combinations. 

The Efforts are on a continuum between two ends:  
Indirect (-)<--> Direct (+)   
Light (-) <--> Strong (+)  
Sustained (-) <--> Sudden (+)  
Free (-) <--> Bound (+)  

Single Effort elements and combinations of all four Efforts are highly unlikely, and they appear only in extreme cases. In our daily lives, we tend to use Effort in combinations of 2 (States) or 3 (Drives). States are more ordinary and common in everyday usage, whereas Drives are reserved for extraordinary moments in life. We have more intense feelings in these distinctive moments, therefore, they convey more information about our personality. With the ultimate goal of developing personality-driven motion synthesis, we focused on States and Drives when designing our user study. Participants were shown a screen of three humanoid mannequins expressing the same action in parallel, but under different effort parameterizations. Participants were tasked with selecting the two motion sequences out of three that they felt were most similar. We focus on States and Drives, plus the neutral expression, in order to systematize the selection of triplets.

### Conda Environment
To install dependencies, cd into the root directory of motion-similarity and create a conda environment from the provided yaml file:
```
$ conda env create -f motion_similarity_env.yml
```
The first line of the yml file sets the new environment's name.
Activate your conda environment by name:
```
$ conda activate motion-similarity
```

### Execute
run the project from the root directory with `python main.py`.

### Reproducing the model sweep
For the full per-model training, embedding-extraction, and evaluation sweep behind the
paper's results (all nine encoders, both raw embeddings and triplet-refined, across walking,
pointing, and picking) — including exact checkpoints, encode commands, environment knobs,
and the geometric baselines — see [docs/REPRODUCING_SWEEP.md](docs/REPRODUCING_SWEEP.md).
The working manuscript draft is at [docs/paper_draft.md](docs/paper_draft.md).




