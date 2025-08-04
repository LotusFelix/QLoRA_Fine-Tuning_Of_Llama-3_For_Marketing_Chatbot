# RESULTS.md

## Logic behind `prepare_data.py` and Instruction Variety

The `prepare_data.py` script handles the transformation of the structured strategic blueprint into five richer, context-aware training instances. Ultimately, the dataset (in addition to the responses from the Re-expression stage) is more contextually encompassing and versatile.

The implementation spans the following steps:
1. **Loads the JSON blueprint** (persona, core insight, persuasion tactic, key message, call to action).  
2. **Defines five channel-specific instructions** that reflect deep marketing psychology and medium nuances.  
3. **Builds `{instruction, context, response}` objects**, writing them as JSONL for downstream fine-tuning.

Also, the strategic approach of using the same detailed context combined with a broader spectrum of diverse instruction formats enriches the model with deeper contextual awareness, enabling it to better generalise across channels.

Take for example, the first instruction **LinkedIn Thought-Leadership Post**: by using this prompt, I present the strategist’s inner conflict by identifying “decision-anxiety,” following this by branding our “Mental Liberation” toolkit as the assured pathway to data-driven confidence. Furthermore, this instruction→context→response combination (especially if we had several types of examples like this) enables the model to learn how to convey authority (further flavoured with empathy) to our target audience. This capability could be monumental in long-form social writing.

Also, in the **150-Word Email Snippet** creation instruction (#Instruction 3), we further resonate the efficacy of our solution with data. Aspects like “slashes anxiety by 50%” teach the model to blend proof with emotional reassurance—vital for high-value direct outreach.

Lastly, in the **LinkedIn InMail** outreach instruction (#Instruction 4), I deliberately leveraged a self-reflective question and an easy, one-click masterclass invite. The response from the re-expression model teaches the model how to better craft conversion-targeted copy, with enhanced capacities in constructing personalised hooks and clear calls to action.

### How a Richer Variety of Instruction Improves Fine-tuning

- **Channel Adaptability:**  
  Furnished with responses for each format, the model learns to generalise the core persuasion tactic across lengths and media.  
- **Emotional Resonance:**  
  The instruction touches on the target persona’s pain point, nudging the model to generate copy that resonates with our audience on both logical and psychological levels.  
- **Format Mastery:**  
  By training on varied response styles, we ensure the fine-tuned model can produce on-brand content in any marketing context.

When you bring these all together, the `prepare_data.py` script produces a deliciously nuanced dataset that better aligns the trained model with our marketer persona’s unique characteristics, spreading through his ambitions, aspirations, and decision triggers. This means the model is fortified with strategic copywriting capability better tailored to the “Constrained Strategist” who is our target.

---

## Single-Day Impact Improvement

Technically, while there is a lot to improve in my fine-tuning implementation, I would say if I had an extra day, I would love to integrate a very specialised human-in-the-loop evaluation cycle alongside automated metrics.

Let us say, after an epoch, I would sample a subset of generated outputs and bring on, say, four seasoned marketers to evaluate the outputs on criteria like **brand alignment**, **emotional resonance**, and **CTA clarity**.

I could take it a step further and automate the collection of such human feedback through a web interface, computing the average scores, which is then fed back into model selection by enabling `load_best_model_at_end=True` with `metric_for_best_model="human_score"`.

This would give the best of both worlds: automated loss/perplexity metrics and a human-driven, nuanced marketing perspective. This way, the QLoRA adapters do more than bring computational optimality (during fine-tuning), but produce improved alignment (output-wise) with the target’s strategic persona.
