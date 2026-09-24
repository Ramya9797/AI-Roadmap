# from llm.chain import create_qa_chain


# def main():

#     chain = create_qa_chain()

#     response = chain.invoke(
#         {
#             "context": """Incident INC0012345 occurred on server APP01. The incident was caused by high CPU utilization.CPU utilization reached 98%. The recommended resolution is to restart the affected application service and investigate the process causing high CPU usage.""",

#             "question": "What database was used?"
#         }
#     )

#     print("\nCHAIN RESPONSE:")
#     print(response.content)


# if __name__ == "__main__":
#     main()

from llm.chain import create_qa_chain


def main():

    chain = create_qa_chain()

    questions = [
        {
            "context": "Customers can request a refund within 30 days.",
            "question": "Can customers request a refund?",
        },
        {
            "context": "Customers can request a refund within 30 days.",
            "question": "How many days do customers have?",
        },
        {
            "context": "Customers can request a refund within 30 days.",
            "question": "What is the company's headquarters?",
        },
        {
            "context": """Incident INC0012345 occurred on server APP01. The incident was caused by high CPU utilization.CPU utilization reached 98%. The recommended resolution is to restart the affected application service and investigate the process causing high CPU usage.""",

            "question": "What caused the incident?"
        },
                {
            "context": """Incident INC0012345 occurred on server APP01. The incident was caused by high CPU utilization.CPU utilization reached 98%. The recommended resolution is to restart the affected application service and investigate the process causing high CPU usage.""",

            "question": "What database was used?"
        }
    ]

    for item in questions:

        response = chain.invoke(item)

        print("\nQUESTION:")
        print(item["question"])

        print("\nANSWER:")
        print(response.content)

        print("\n" + "-" * 50)


if __name__ == "__main__":
    main()