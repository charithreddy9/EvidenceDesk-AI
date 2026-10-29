"""Separate *author-held-out* question set for staging regression.

These question strings were not used in the original 30-case dashboard or
the threshold design. They are still manually authored over the same four
fictional documents, not an external independent benchmark.
"""
from .evaluation_report import build_report

HOLDOUT_CASES = [
    {"id":"h01","category":"direct","question":"What allowance is provided for meals each day?","expected_document":"policy-travel","expected_fragment":"$65 per day"},
    {"id":"h02","category":"direct","question":"What timeframe applies to receipt submission?","expected_document":"policy-travel","expected_fragment":"30 days"},
    {"id":"h03","category":"direct","question":"Where should secret production credentials live?","expected_document":"policy-security","expected_fragment":"secrets manager"},
    {"id":"h04","category":"direct","question":"What is the diagnostic log retention period?","expected_document":"policy-data","expected_fragment":"90 days"},
    {"id":"h05","category":"paraphrase","question":"Which vault holds login keys for live systems?","expected_document":"policy-security","expected_fragment":"secrets manager"},
    {"id":"h06","category":"paraphrase","question":"What is the deadline for an outage write-up?","expected_document":"policy-incident","expected_fragment":"five business days"},
    {"id":"h07","category":"paraphrase","question":"What are the rules on copying client information into public AI?","expected_document":"policy-data","expected_fragment":"must not copy customer data"},
    {"id":"h08","category":"paraphrase","question":"When should a critical incident be acknowledged?","expected_document":"policy-incident","expected_fragment":"ten minutes"},
    {"id":"h09","category":"unanswerable","question":"How are annual bonuses calculated?","expected_document":None,"expected_fragment":None},
    {"id":"h10","category":"unanswerable","question":"Is bicycle parking free for staff?","expected_document":None,"expected_fragment":None},
    {"id":"h11","category":"unanswerable","question":"Which fitness classes are covered by employee benefits?","expected_document":None,"expected_fragment":None},
    {"id":"h12","category":"unanswerable","question":"How long is the paid maternity leave allowance?","expected_document":None,"expected_fragment":None},
    {"id":"h13","category":"unanswerable","question":"How much is the home internet stipend?","expected_document":None,"expected_fragment":None},
    {"id":"h14","category":"unanswerable","question":"When do restricted stock units vest?","expected_document":None,"expected_fragment":None},
]
def compare_holdout(index):
    after=build_report(index,HOLDOUT_CASES)
    before=build_report(index,HOLDOUT_CASES,verifier=False)
    return {"baseline_summary":before["summary"],"verified_summary":after["summary"],
            "comparison":{k:round(after["summary"][k]-before["summary"][k],4) for k in
                ("hit_at_1","recall_at_3","mrr","abstention_accuracy","case_pass_rate")},
            "cases":after["cases"]}
