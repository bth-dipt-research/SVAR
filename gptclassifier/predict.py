from pathlib import Path
from openai import OpenAI
from keys import OPENAI_KEY

import pandas as pd
import json
import sys
import argparse

from utils import Classification

instructions = """
You are an expert in requirements engineering and construction (roads, railways, infrastructure). Your task is to classify infrastructure requirements written in Swedish to identify unverifiable requirements. The classification follows a specific scheme, considering four dimensions with the labels in square brackets: target [Product, Process, Documentation], nature [Quantitative, Qualitative, Mixed], interpretability [Non-ambiguous, Ambiguous (natural), Ambiguous (artificial)], and reference [Local, Internal, External, No reference]. You classify a requirement based on the instructions and definitions below, and output solely the label for each dimension.

Consider the following points when performing the classification:
Assume the perspective of someone responsible for verification. This is important for the correct classification of interpretability. A requirement may be ambiguous from the perspective of a designer who needs to develop a solution that satisfies a requirement. Such a requirement can be, as we defined it, naturally ambiguous, i.e. the requirements writer left the solution open. Such a requirement can be still verifiable, i.e. it is possible to determine a mechanism that checks if the requirement is fulfilled. There are requirements that are solution-closed, i.e. that specify how the solution must look like. In the strict sense, this is not a requirement but rather a specification. Such a solution-closed requirement is not naturally ambiguous and therefore, if it does not contain any other form of ambiguity, implementable and verifiable. However, our focus activity is verification and not implementation.

Sometimes, a requirement may specify under which circumstances it applies. For example:
- For welding gap 68 mm, thermite form L68 must be used.
- Foundations that are not protected by road railings or the like must be placed so that no part of the foundation's upper surface is located higher than 0.1 m above finished ground.

In these cases, the nature of the requirement (quantitative, qualitative, mixed) should be determined by the content of the requirement, not the condition in which the requirement applies.
References do not need to refer to concrete artifacts, but can point to information in a general way. Example:
- K46926: Steel doors must be painted light gray according to the manufacturer's standard.
All dimensions must be classified, independently of the target of the requirement (process or documentation).

Target
The target of a requirement is the general field to which the requirement applies. It is important to differentiate between the targets as the type of verification on whether the requirement is fulfilled can vary depending on the field.

A product requirement refers to an artifact delivered from a supplier to the client. This artifact can be digital or physical. A process requirement specifies how an artifact shall be produced, i.e. it defines possibilities and limitations of the process that the supplier shall follow to create and deliver the product. There exist requirements that are formulated in a way that suggests they are process requirements (e.g. “X needs to be designed to achieve Y”, or “X needs to be placed such that Y”). The formulation refers to the design process or production process. However, these requirements intrinsically refer to properties/characteristics of the product and possibly its environment. Hence, we classify these requirements as product requirements.   A documentation requirement specifies what information shall be recorded, either for the process or the product. For digital products, this can be requirements on what and how the information is delivered from a supplier to a client. For physical products, this can be requirements on what information is required for maintenance for the products.

Nature
The type of information provided in the requirement can be either qualitative, quantitative or a mixture of both. Quantitative information contains numerical values or anything that can be objectively interpreted as a numerical value. Anything else is qualitative. A mixture of qualitative and quantitative information is also possible.

Interpretability
Identifies whether the requirement is objective or subjective. An objective requirement leads to one, and only one, interpretation what requirement fulfillment means. A subjective requirement can be interpreted in different ways, by different people. Subjectivity can be differentiated in natural and artificial ambiguity.
Natural ambiguity is intended ambiguity to allow for design (solution) flexibility. Performance requirements, i.e. rules that require a “proof of solution”  are typically requirements that are formulated with natural ambiguity. Natural ambiguity is a decision, i.e. regulators chose to include a certain level of ambiguity, either to promote design flexibility or independence of technological changes.
Artificial ambiguity is introduced involuntarily (due to lack of expertise in formulating requirements) and can be eliminated by better use of objective, precise, clear, non-redundant and independent sentences.
Sources of artificial ambiguity are:
1) Use of language
- Vagueness: Vagueness often results from poor or sloppy use of language. In building requirements, it typically appears in insufficiently defined words or phrases, often including adjectives or adverbs (e.g., long enough, sufficient).
- Incompleteness: Incompleteness refers to cases when there is missing information in the rule provision. Because of the missing information, there is more than one possible interpretation of the rule provision.
- Lexical ambiguity: Lexical ambiguity is caused by a word or phrase having multiple meanings. A sentence is ambiguous when these meanings are all plausible and the reader cannot tell which meaning is intended by the regulator.
2) Tacit knowledge
Building requirements are written to summarise and convey engineering and construction knowledge. With careful writing, explicit engineering and construction knowledge can be presented unambiguously. However, apart from that, a large amount of tacit knowledge cannot be easily formalised, aggregated or written down.

Reference
A reference points to information that is not stated within the scope of the requirement. A reference can be local, internal or external.
A local reference points to a location within the same document. For example a requirement can refer to a figure, a table or an appendix.
An internal reference points to a document within the same collection of documents. For example, a requirement can refer to another requirement or a figure in another document. Internal means all documents that are under the control of the same organization.
An external reference points to a document that is not under the control of the organization that owns the requirement. For example, a requirement can point to a standard issued by a standardization body or a law issued by a government.
"""

def classify(**kwargs):
    results = []
    for index, row in ground_truth.iterrows():
        id = row['ID']
        requirement = row['requirement']

        messages = [
            {'role': 'system', 'content': instructions},
            {'role': 'user', 'content': requirement}
        ]

        response = client.responses.parse(model=modelname,
                                          input=messages,
                                          text_format=Classification,
                                          **kwargs
                                          )

        classification_data = response.output_parsed
        print(f'{modelname} ({index + 1}/{samples}):{requirement} --> {classification_data}')

        results.append({
            'id': id,
            'requirement': requirement,
            'classification': classification_data.model_dump()
        })

    with open(outputdir / f'predictions_{modelname}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', type=str, help='Test set file')
    parser.add_argument('modelname', type=str, help='Name of the GPT model')
    parser.add_argument('outputdir', type=str, help='Directory to write the predictions')

    args = parser.parse_args()
    modelname = args.modelname
    inputfile = args.inputfile
    outputdir = Path(args.outputdir)

    models = ['gpt-4o-mini', 'gpt-4o', 'gpt-4.1']
    reasoning_models = ['o4-mini', 'o3', 'gpt-5-nano', 'gpt-5']

    if modelname in models:
        is_reasoning_model = False
    elif modelname in reasoning_models:
        is_reasoning_model = True
    else:
        print('Unknown model specified. Exiting...')
        sys.exit(1)

    ground_truth = pd.read_csv(inputfile, encoding='latin-1')
    samples = len(ground_truth)

    client = OpenAI(api_key=OPENAI_KEY)

    if is_reasoning_model:
        classify(reasoning={'effort': 'medium'})
    else:
        classify(temperature=0)





