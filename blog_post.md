---
date:
  created: 2026-05-19
categories:
    - LLM
authors:
    - ltdarc
---

# Introduction

Designing LLM Evaluation Frameworks may sound niche but it's actually an essential part of any AI workflow, especially for data extraction tasks. A robust framework allows researchers to quickly get a sense of how different models perform across a set of sample images.

This article will cover the thought process behind how we designed our own LLM Evaluation Framework, the results, and learnings from this project. To recreate and customize your own framework you can find our github here ( ** insert link here ** ).

# Data Extraction is Time Consuming

For social science or business research data often comes from dense tables in PDFs. Some examples might be city council meeting notes, SEC filings, or TV guides like the one below.

** insert image here **

In the past these images might get outsourced to a third party or a student with instructions to go cell by cell and transcribe the all of the information into a google sheets document. Later a supervisor may check the outputs to help minimize errors. Overall this is a pretty time intensive and manual task.

# Exploring AI Alternatives

LLM's are powerful, cutting edge tools that have a wide array of applications. They may seem like an easier alternative to manual data extraction but I would argue that deciding whether or not incoporate them into a workflow is a deceptively tough question. 

** insert image here **

That's because results can be entirely dependent on which LLM you're planning on using. Which is then dependent on what the given task is, what data we're extracting information from, how accurate the results need to be, and what budget is available.

As a first step you might log into the Stanford AI Playground ( ** insert link here ** ). You can upload images and add description of what you want the model to do, after a few seconds the model should give you an output.

** insert gif here **

But if the goal is to find the best model for the task we should test all multimodal LLMs available, which would over 20 at the time of this post being published. Using just 1 image likely isn't enough and we'd typically want multiple different pieces of information from each document.

** insert image here **

35 images, 6 benchmarks (unique outputs), and 18 models quickly scale to just shy of 3,800 unique combinations. This isn't feasible to feed into the Stanford AI Playground one by one, so how can leverage LLMs at scale?

# Building Pipelines

This is where setting up an automated pipeline becomes crucial. You can find a visual representation of my project pipeline below, which processes a single task. We used the Yens for compute and SLURM array jobs helped us process tasks in parallel.

** insert image here **

## Selecting Inputs

Benchmarks: these are what you want the model to do/what type of information the model should extract. This will include prompts, what type of output we're looking for (string, array, etc.), and the metric we're using to evaluate accuracy.

** include screenshot of benchmark JSON **

Images: where the LLM will extract data from. We did some preprocessing through converting color PDFs into greyscale PNGs to ensure we were able to fit within the context limits of each model.

Models: which LLMS are doing the extraction.

## Extraction

LLMs on the Stanford AI Playground can be accessed on the Yens via the API ( ** insert link here ** ). You will need to apply and get approval for an API key.

Once inputs have been selected we can feed the prompts and images into the models we've chosen via the API.

## Evalution

Model outputs are written directly to the MongoDB database which acts as a central information store that allows to easily query results. In the earlier example we mentioned needing a supervisor to check the results of our human transcriber. Similarly, we also need to build an evaluation component into this pipeline because LLMs can be prone to errors and hallucinations.

## Results

Due to the design choices that we made the pipeline is able to finish processing and evaluating all 3,780 tasks in just a few hours. 

# Benchmarks

For every TV Guide that was used in our pipeline we had a corresponding ground truth document that was transcribed by a person. When selecting benchmarks we wanted to choose outputs that could be found within the ground truth, i.e. they have to be definitive vs subjective. We started with 6 benchmarks intially:

** include image **

** include image **

Easy (Grey):
- Simple metadata extraction tasks
- Same location across documents, high resolution

- Newpaper Name
- Newspaper Date

Medium (Yellow):
- TV Guide Day Of Week
    - Location can change across image
    - Closer to actual table, mixed resolutions

- TV Guide Date
    - Reasoning: LLM needs to combine Newspaper Date and TV Guide Day of Week without being explicitly told to do so

Hard (Grid):
- Data is found within the grid itself
- Smallest font, lowest resolution
- Variation in placement across images

- First channel
- first program

# Initial Results

## Results across Images

Looking at the average accuracy score by image for all models and benchmarks showed a pretty wide range of results.  

** include image **

The best image (#22) had a 40 pt. difference compared to our worst image (#23). We've included images of both below, can you guess which one is which?

** include image **

Image 22 is on the left and image 23 is on the right. Was that what you guessed? 

For images that seem pretty similar at first glance they had drastically different results, showing that how LLMs understand images remains a bit of a black box.

## Results by Model

** include image **

Across all images an benchmarks our best performing model was gemini-2.5-pro with 72% accuracy and a total token cost of $8.76. Our worst performing model was claude-3-haiku with 51% accuracy and a total token cost of $0.35. 

Most other models in playground fell somewhere between the two in both accuracy and cost. The only exception would be o1 which was the most expensive model with a total cost of $41.35. Based on these results we would recommend avoiding this image for tabular data extraction.

## Results by Benchmark

Looking at benchmark accuracy rates across all images and models, the results seemed to confirm the difficulty rating assigned to them.

** include image **

Newspaper name and newspaper date had the highest accuracy while first channel and first program were accurate less than one third of the time.

## Selecting LLMs based on Benchmark

One thing we wanted to caution against is assuming that one model will preform the best across all tasks. The below shows the best model for each benchmark in terms of highest accuracy and lowest price.

**  include image **

As you can see the supposed worse model, claude-3-haiku, was actually the best model for two of these benchmarks. The reason for this is that most of the LLMs performed similarly for the same prompt but claude-3-haiku was more efficient in token usage which results in lower costs.

## First Program and First Channel

Given that first program and first channel were our two worst performing benchmarks I wanted to double click and look a per model performance across the entire image set.

** include image **

While Llama-4 and gemini-2.5-pro were the best performing models for these benchmarks even they couldn't get above 60% accuracy. This would be an unacceptable accuracy rate for reserach data which leads to our next set of questions. Are the models just bad? Or is there an opportunity for us to further optimize results?

# Revisiting our benchmarks

** include image **

Going back to our earlier questions we can use them to help identify areas for improvment. Let's start with task. 

How well is what we're trying to do being reflect in the prompt we're giving the model?

First Program User Prompt (v1): "Return the name of the program for the first channel listed and for the earliest time slot shown."

This prompt we intially used is only one sentence. It feels pretty vague and doesn't give all that much guidance.

First Program User Prompt (v2): "Analyze the provided image of a TV schedule grid. Channels are typically listed vertically (rows) and time slots horizontally (columns). Your task is to extract the program title for the FIRST channel listed at the EARLIEST time slot shown. Follow these steps carefully: 1. Scan the grid to identify the top-most row containing programming data (the row immediately below the time-slot or any other subsection headers). 2. Scan to the left-most time block within that specific row. 3. Identify the text inside this top-leftmost program block.  4. Transcribe the text exactly as printed. Include all numbers (e.g., episode numbers, parts, movie years), abbreviations, and characters that appear immediately with the title."

In our second iteration we looked at the instructions we'd previously given the third party transcription service and used it as a starting point. The prompt now gives the model explicity instructions to think about the image as a grid with rows and columns. It also gives clear guidance on where the key information is located within the grid.

First Program User Prompt (v3): "Analyze the provided image of a TV schedule grid. Channels are typically listed vertically (rows) and time slots horizontally (columns). Your task is to extract the program title for the FIRST channel listed at the EARLIEST time slot shown. Follow these steps carefully: 1. Scan the grid to identify the top-most row containing programming data (the row immediately below the time-slot or any other subsection headers). 2. Scan to the left-most time block within that specific row. 3. Identify the text inside this top-leftmost program block.  4. Return only the title, ignore all closed captioning markers, rerun indicators, movie release years, or VCR Plus+ codes (numeric sequences) that appear immediately with the title."

In our third iteration we adusted the last line of our prompt to focus on the name itself and ignore all other miscellanious information. This better reflected what an actual research question might be and helped narrow the focus of the LLM.

Once we feel confident that our tasks are better explained in the prompt we can move on to the next step.

# Taking another look at our data

** include image **

When we think about improving a score it makes sense to review our responses. But should we also review the ground truth that we're comparing our responses against?

** insert image **

For the above image what do you think is the right output for first program?

A. 2015 Daytona 500 The 57th running of the event. The race consists of 200 laps and is the first race of the season. (N) (cc)

B. 2015 Daytona 500 The 57th running of the event. The race consists of 200 laps and is the first race of the season.

C. 2015 Daytona 500

Correct answer: it depends.

Hmm, let's try another one. For the below image what should the first channel output be?

** insert image **

A. 2 3 003 2 KHON

B. 2 3 003 2

C. KHON

Correct answer: it also depends.

For a seemingly simple questions there's can be a lot of different "right" answers. The ground truth is dependent on the research question that's being answered. Are we looking at how many programs offered closed captions? Do we care about about all the different channel numbers associated with a network?

One of the best ways to get a better grasp of your data is to actually hand transcribe 5 to 10 images. This will help you get a sense of what data you have available and how much it can vary across your dataset.

# Choosing metrics

In the first part of our project we evaluated results via exact matching. If there weren't more or less exactly what we expected the ground truth to be the LLM would score a 0. But this doesn't help us measure how close an output is to the ground truth. And should we be using the same metric even if the output types are different?

** insert image **

When redesigning our LLM workflow our though process was to first start with what research question we were trying to answer. In order to answer the question we would need to complete a series of tasks. Prompts helped us translate these tasks into instructions for an LLM to follow. Finally, the metric should be based on what the prompt is asking the model to do. 

In this way if (1) the task reflects the research question, (2) the prompt reflects and task, and (3) the metrics reflect the prompt then the metric becomes an import signal for the workfow.

** insert image **

If the metric suggest poor performance it may be worth questioning:

1. Are my tasks reflective of my research question and the data I have to work with?

2. Does my prompt properly explain what I want the LLM to do?

3. Is my metric appropriate for the actual prompt?

My using our metrics as a signal we iterated through this thought process several times to refine our outputs.

** insert image **

First Program and First Channel were our lowest performing benchmarks so we focused on them when trying to optimize the quality of our outputs. As we refined our prompts the model output type changed as well which caused us to update our metrics as well (ex. calculating set overlap if the prompt asked the model to return an array of channels).

We also added two new benchmarks, All Times and All Channels, to test how well the models were capable of extracting larger arrays of data.

# Updated Results




