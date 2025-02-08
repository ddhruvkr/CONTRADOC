insert_start_phrase = '''
In the given article, your task is to identify 5 factual statements which objectively describe an individual or a situation, with no relation to subjective matters such as opinions, emotions, preferences, attitudes or feelings. Each statement should present a concrete fact about the subject in question.
After identifying these statements, formulate a contradictory statement for each one. These contradictions must indicate a scenario which cannot coexist with the original statements, thereby presenting mutually exclusive facts. Remember, the goal is not to negate the original statement but to offer a fact that renders the original fact impossible.
You are allowed to omit some irrelevant details or make up things. The following examples can guide you:
Good Examples:
Original: Jane and her sister has a good relationship.
Contradiction: Jane is an only child.

Original: Tim Johnson, ... He was born in the UK and moved to Mexico with his family in 2006.
Contradiction: Tim Johnson was born in Paris.

Original: A penguin is a bird.
Contradiction: A penguin is a mammal.

Original: Tom's wife, Lisa, is an accountant and she earns much more than Tom.
Contradiction: Lisa is Tom's sister.

Bad Example:
Original: Lisa is a professor.
Contradiction: The university's headmaster is Lisa.
(This is a poor example since one can simultaneously be a professor and a headmaster, therefore, these are not mutually exclusive scenarios.)

Your contradiction statements should meet requirments below:
1. Directly contradict the original factual statement.
2. Be focused on factual discrepancies rather than emotional or subjective ones.
3. Is independently reasonable and sounds natural in the article, not merely a negation of the original statement.
4. Utilize different wording from the original statement, keeping word overlap to a minimum, rephrase it is possible.
5. Use the original names of entities rather than pronouns.
Please respond in JSON format as a list of 5 elements, each element structured as follows: {"statement": "YOUR_SELECT_STATEMENT_1", "contradiction":"CONTRADICTION_TO_STATEMENT_1"}.
Article:
'''


replace_start_phrase = '''
In the given article, your task is to identify 10 factual statements which objectively describe an individual or a situation, and rewrite the sentence in the article to be a contradiction of the statement you select. It can be either negation or making up some facts that are mutually exclusive with the original statament.
Good Examples:
Original Sentence: National Hurricane Center director Bill Proenza left his position Monday, just days after nearly half of the NHC staff signed a petition calling for his ouster. 
Select Statement: National Hurricane Center director Bill Proenza left his position Monday.
Contradicted Statement: Bill Proenza remained a director of National Hurricane Center.
Rewrited Sentence: Bill Proenza remained a director of National Hurricane Center, just days after nearly half of the NHC staff signed a petition calling for his ouster. 

Original Sentence: Tim Johnson, is one of the most famous artists in the world, who was born in the UK and moved to Mexico with his family in 2006.
Select Statement: Tim Johnson was born in UK.
Contradicted Statement: Tim Johnson was born in Paris.
Rewrited Sentence: Tim Johnson, is one of the most famous artists in the world, who was born in Paris and moved to Mexico with his family in 2006.

Original Sentence: Hurricane center staffers told CNN's John Zarella they were unhappy not only about his comments about the QuikSCAT, but also about the environment at the center.
Select Statement: Hurricane center staffers were unhappy about his comments.
Contradicted Statement: Hurricane center staffers were happy about his comments and the environment at the center.
Rewrited Sentence: Hurricane center staffers told CNN's John Zarella they were happy not only about his comments about the QuikSCAT, but also about the environment at the center.

Please respond in JSON format as a list of 10 elements, each element structured as follows: {"Original Sentence": "SENTENCE_FROM_ARTICLE", "Select Statement":"ONE_STATEMENT_FROM_SENTENCE", "Contradicted Statement": "Contra_ONE_STATEMENT_FROM_SENTENCE", "Rewrited Sentence":"MODIFIED_SENTENCE_FROM_ARTICLE"}
Article:
'''

subjective_start_phrase_replace = '''
In the given article, your task is to identify 3 subjective statements throughout the article which describe one's emotions/moods/opinions/views/preferences, and give a contradiction of the statement you select. It can be making up some facts that don't fit the context. Please avoid simply negating the original statement by adding 'not'.

Examples:

E.g., The article introduces a scientific breakthrough and an expert expressed his feeling: "This research is significant and it can change life of millions of people." , Change to an expert expressed his feeling: "This research has no impact at all."
E.g., The article is about searching for the missing student. And change "The rescue team searched for the boy worriedly." to "The rescue team searched for the boy happily."
E.g., The article is about some workers who fought against the company for their rights, Change "When the court ruled that the company didn't need to pay compensation, all workers burst into rage." into "When the court ruled that the company didn't need to pay compensation, all workers cheered up from their seats. "

Out of all statements in the article, you can pick 3 statements that are most natural to contradict. Please respond in JSON format as a list of 3 elements, each element structured as follows: {"Original Sentence": "SENTENCE_FROM_ARTICLE", "Select Statement":"IN_YOUR_OWN_WORDS", "Contradicted Statement": "Contra_SELECT_STATEMENT", "Rewritten Sentence":"MODIFIED_SENTENCE_FROM_ARTICLE"}
Article:
'''