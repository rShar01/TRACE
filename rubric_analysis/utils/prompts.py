refined_proposer_axis = """
You are a machine learning researcher analyzing two large language models (LLMs) by comparing how their responses differ to the same set of questions. Your goal is to identify unique, interpretable behavioral dimensions ("axes of variation") that capture subtle or surprising differences between the models.

Here are the questions and responses:
{combined_responses}

For each axis, describe what makes one model's responses higher and the other's lower on that dimension. Focus on differences that reveal deeper behavioral tendencies rather than surface traits.

Format your output as a bulleted list, with each axis on a new line starting with a dash (-) or asterisk (*). Each axis should follow this format:

- {{axis}}: High → {{description of high end}} | Low → {{description of low end}}

Example:
- Self-consistency: High → Responses maintain consistent reasoning throughout | Low → Reasoning may shift or contradict earlier statements

Guidelines:
- Avoid obvious or generic dimensions such as "clarity," "conciseness," or "formality."
- Look for behavioral nuances that might emerge from reasoning patterns, goal orientation, implicit assumptions, moral framing, creativity style, uncertainty handling, or tone of confidence.
- Axes can mix abstract and domain-specific aspects (e.g., "self-consistency," "risk aversion in code design," "emotional attunement to user intent," "degree of self-critique," "literalness vs. interpretation").
- Each axis must be something a human could use to categorize which model response is higher or lower.
- Do not add explanations, prefaces, or summaries.
- If you find no new substantive differences, output only “No differences found.”
"""

refined_proposer_axis_iteration = """
You are a machine learning researcher analyzing how two large language models (LLMs) differ in their responses to the same set of questions, and whether these behavioral differences may correspond to user preferences. Some differences have already been identified, but there are more subtle variations still to find. Your task is to propose additional axes of variation that are not already covered in the existing list.

For each new axis, describe an interpretable behavioral property that a human could use to categorize one response as higher or lower on that axis. Focus on *novel and meaningful* distinctions, not generic traits.

Here are the existing differences, followed by the questions and responses:
{combined_responses}

Format your output as a bulleted list, with each axis on a new line starting with a dash (-) or asterisk (*). Each axis should follow this format:

- {{axis}}: High → {{description of high end}} | Low → {{description of low end}}

Example:
- Refactoring instinct: High → Code responses show tendency to restructure and optimize | Low → Code responses focus on direct implementation without optimization

Guidelines:
- Do **not** repeat any existing axes or restate similar concepts.
- Avoid generic or overused axes such as "conciseness," "clarity," or "formality."
- Seek human-interpretable, behaviorally rich dimensions that capture tendencies like reasoning style, emotional framing, abstraction level, self-consistency, precision, or ethical orientation.
- Consider both general behaviors and domain-specific traits (e.g., “refactoring instinct in code,” “imaginative elaboration in storytelling,” “error tolerance in math reasoning”).
- Do **not** reference Model 1 or Model 2 by name.
- Do **not** include explanations, summaries, or commentary outside the list.
- If you find no new substantive differences, output only “No differences found.”
"""

proposer_freeform_axis = """You are a machine learning researcher trying to figure out the major differences between the behaviors of two llms by finding differences in their responses to the same set of questions. Write down as many differences as you can find between the two outputs. Please format your differences as a list of axes of variation and differences between the two outputs. Try to give axes which represent a property that a human could easily interpret and they could categorize a pair of text outputs as higher or lower on that specific axis. 

Here are the questions and responses:
{combined_responses}

The format should be a list of axes in the format of {{axis}}: High: {{high description}} Low: {{low description}} for each axis, with each axis on a new line separated by *. Do NOT include any other text in your response.

Consider differences on many different axes such as tone, language, structure, content, safety, and any other axis that you can think of. If the questions have a specific property or cover a specific topic (e.g. coding, creative writing, math, etc.), also consider differences which are relevant to that property or topic.
If there are no substantive differences between the outputs, please respond with only "No differences found."
"""

proposer_freeform_iteration_axis = """You are a machine learning researcher trying to figure out the major differences between the behaviors of two llms by finding differences in their responses to the same set of questions and seeing if these differences correspond with user preferences. I have already found some differences between the two outputs, but there are many more differences to find. Write down as many differences as you can find between the two outputs which are not already in the list of differences. Please format your differences as a list of properties that appear more in one output than the other.

Below are multiple sets of questions and responses, separated by dashed lines. For each set, analyze the differences between Model 1 and Model 2. Please format your differences as a list of axes of variation and differences between the two outputs. Try to give axes which represent a property that a human could easily interpret and they could categorize a pair of text outputs as higher or lower on that specific axis. 

Here are the differences I have already found and the questions and responses:

{combined_responses}

The format should be a list of axes in the format of {{axis}}: High: {{high description}} Low: {{low description}} for each axis, with each axis on a new line separated by *. Do NOT include any other text in your response.

Consider differences on many different axes such as tone, language, structure, content, safety, and any other axis that you can think of. If the questions have a specific property or cover a specific topic (e.g. coding, creative writing, math, etc.), also consider differences which are relevant to that property or topic.
    
Remember that these differences should be human interpretable and that the differences should be concise, substantive and objective. Write down as many properties as you can find which are not already represented in the list of differences. Do not explain which model has which property, simply describe the property. Your response should not include any mention of Model 1 or Model 2.
Respond with a list of new properties, each on a new line separated by *. Do NOT include any other text in your response. If there are no substantive differences between the outputs, please respond with only "No differences found."
"""

annotation_comment_proposer = """
You are a machine learning researcher analyzing annotator comments to surface unique, interpretable behavioral dimensions ("axes of variation") that capture what annotators notice when preferring one answer over another. Work only from the comments—do not assume anything about the original questions or answers.

Here are the comments to analyze:
{comments}

For each axis, describe what makes a response higher versus lower on that dimension. Focus on differences that reveal deeper behavioral tendencies rather than surface traits.

Format your output as a bulleted list, with each axis on a new line starting with a dash (-) or asterisk (*). Each axis should follow this format:

- {{axis}}: High → {{description of high end}} | Low → {{description of low end}}

Guidelines:
- Derive axes only from the themes present in the comments (e.g., syntax validity, conciseness, unnecessary extras, instruction alignment).
- Look for interpretable, discriminative properties (reasoning patterns, goal orientation, adherence to constraints) rather than generic “good/bad.”
- Keep axes human-usable; a reviewer should be able to place an answer as higher or lower on the axis from the comment.
- Do not mention specific questions, models, or options—focus on underlying properties.
- If no substantive differences are present, output only “No differences found.”
"""

reduce_freeform_axis = """The following are axes of variation that can be used to compare two model outputs, each with a name and a description of what makes an output high or low on that axis. Some axes may be redundant, misnamed, or overlap with others. Your goal is to cluster and reduce these axes into a minimal set of parent axes that are as *distinct and non-overlapping* as possible, while still preserving the *specificity and uniqueness* found in the original refined axes—don't over-merge unique properties. 

For each new parent axis you create:
- Ensure the high/low descriptions clearly encompass and are faithful to the axes they subsume, but retain distinctive properties rather than over-generalizing. 
- If any axis is truly unique or nuanced, keep it as its own parent axis rather than forcing it to merge.
- The parent axes should be mutually exclusive and make it easy for a human to reliably and uniquely categorize model outputs along each axis.
- If an axis is specific to a domain or task (e.g., coding), make sure its name reflects this specificity.

Here are the axes of variation (each formatted as {{axis name}}: High: {{high description}} Low: {{low description}}):
{differences}

Cluster and reduce these axes into a minimal, clear set of parent axes (retaining uniqueness when present). Each parent axis should include an axis name and a concise (<20 words) description, preserving any domain-specific or unique distinctions in the original. 

Format your output as a bulleted list, one axis per line, each starting with a dash (-) or asterisk (*), with this format:

- {{axis}}: High → {{description of high end}} | Low → {{description of low end}}
"""

ranker_prompt_multi = """You are a fair and unbiased judge. Your task is to compare the outputs of two language models (A and B) on a set of one or more properties. Which response better aligns more with each property, A, B, or equal?

Your sole focus is to determine which response better aligns with each property, NOT how good or bad the response is. Consider each property independently. Do NOT let the position of the model outputs influence your decision and remain as objective as possible. Consider what each property means and how it applies to the outputs. Would a reasonable person be able to tell which output aligns more with the property based on the description?

Instructions: For each property,
	•	If Response A aligns with the property more than Response B, respond with "A".
  •	If Response B aligns with the property more than Response A, respond with "B".
	•	If the responses are roughly equal on the property or neither response contains the property, respond with "equal". 
	•	If the property does not apply to these outputs (e.g., the property is about code quality, but the prompt is not related to coding), respond with "N/A".
	•	If you are unsure about the meaning of the property, respond with "unsure". Think about of a reasonable person would find the property easy to understand.

A group of humans should agree with your decision. The properties will be given in the form of a numbered list. Use the following format for your response:
Ranking:
Property 1: {{A, B, equal, N/A, or unsure}}
Analysis: {{reasoning}}
Property 2: {{A, B, equal, N/A, or unsure}}
Analysis: {{reasoning}}
Property 3: {{A, B, equal, N/A, or unsure}}
Analysis: {{reasoning}}
...

Here are the properties and the two responses:
{inputs}

Remember to be as objective as possible and strictly adhere to the response format. You must give a ranking for each property."""
