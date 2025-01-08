import argilla as rg

__questions = [
    rg.LabelQuestion(
        name="target",
        title="Target",
        labels=["Product", "Process", "Documentation", "Mixture"],
        description="A product requirement refers to an artifact delivered from a supplier to the client. This artifact can be digital or physical.

A process requirement specifies how an artifact shall be produced, i.e. it defines possibilities and limitations of the process that the supplier shall follow to create and deliver the product.

A documentation requirement specifies what information shall be recorded, either for the process or the product. For digital products, this can be requirements on what and how the information is delivered from a supplier to a client. For physical products, this can be requirements on what information is required for maintenance for the products.",
        required=True),
    rg.LabelQuestion(
        name="nature",
        title="Nature",
        labels=["Qualitative", "Quantitative", "Mixture"],
        description="The type of information provided in the requirement can be either qualitative, quantitative or a mixture of both. Quantitative information contains numerical values or anything that can be objectively interpreted as a numerical value. Anything else is qualitative. A mixture of qualitative and quantitative information is also possible.",
        required=True),
    rg.LabelQuestion(
        name="interpretability",
        title="Interpretability",
        labels=["Non-ambiguous", "Ambiguous (natural)", "Ambiguous (artificial)"],
        description="Interpretability describes whether the requirement is objective or subjective. An objective requirement leads to one, and only one, interpretation what requirement fulfillment means. A subjective requirement can be interpreted in different ways, by different people. Subjectivity can be differentiated in natural and artificial ambiguity. Natural ambiguity is intended ambiguity to allow for design (solution) flexibility. Performance requirements, i.e. rules that require a “proof of solution” are typically requirements that are formulated with natural ambiguity.

Artificial ambiguity is introduced involuntarily (due to lack of expertise in formulating requirements) and can be eliminated by better use of objective, precise, clear, non-redundant and independent sentences.",
        required=True),
    rg.LabelQuestion(
        name="reference",
        title="Reference",
        labels=["None", "Local", "Internal", "External"],
        description="A reference points to information that is not stated within the scope of the requirement. A reference can be local, internal or external.

A local reference points to a location within the same document. For example a requirement can refer to a figure, a table or an appendix.

An internal reference points to a document within the same collection of documents. For example, a requirement can refer to another requirement or a figure in another document. Internal means all documents that are under the control of the same organization.

An external reference points to a document that is not under the control of the organization that owns the requirement. For example, a requirement can point to a standard issued by a standardization body or a law issued by a government.",
        required=True),
    rg.LabelQuestion(
        name="logicrule",
        title="Logic rule",
        labels=["Yes", "No"],
        required=True)
]

__fields = [
    rg.TextField(name="requirement", required=True)
]

dataset = rg.FeedbackDataset(
    fields = __fields,
    questions = __questions
)
