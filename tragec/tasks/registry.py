from tragec.tasks.task_mlm import create_mlm_model
from tragec.tasks.task_mrm import create_mrm_model
from tragec.tasks.task_multiclass import create_multiclass_model
from tragec.tasks.task_singleclass import create_seqclass_model
from tragec.tasks.task_valueprediction import create_valuepred_model
from tragec.tasks.task_pairwisecontact import create_contactpred_model
from tragec.tasks.task_seq2seqclass import create_seq2seqclass_model


def create_and_register_models(namespace, cls, gec_model, prot_model, name):
    if gec_model:
        gec_base_name = gec_model.__name__.replace('Model', '')
        namespace[f"{gec_base_name}ForMaskedRecon"] = create_mrm_model(cls, gec_model, name, 'gec')
        namespace[f"{gec_base_name}ForSequenceClassification"] = create_seqclass_model(cls, gec_model, name, 'gec')
        namespace[f"{gec_base_name}ForMultilabelClassification"] = create_multiclass_model(cls, prot_model, name, 'gec')

    if prot_model:
        prot_base_name = prot_model.__name__.replace('Model', '')
        namespace[f"{prot_base_name}ForMLM"] = create_mlm_model(cls, prot_model, name, 'prot')
        namespace[f"{prot_base_name}ForSequenceClassification"] = create_seqclass_model(cls, prot_model, name, 'prot')
        namespace[f"{prot_base_name}ForMultilabelClassification"] = create_multiclass_model(cls, prot_model, name,
                                                                                            'prot')
        namespace[f"{prot_base_name}ForValuePrediction"] = create_valuepred_model(cls, prot_model, name, 'prot')
        namespace[f"{prot_base_name}ForSeq2SeqClassification"] = create_seq2seqclass_model(cls, prot_model, name,
                                                                                           'prot')
        # Contact prediction is current non-functional because of prediction/target size mismatch
        # namespace[f"{prot_base_name}ForContactPrediction"] = create_contactpred_model(cls, prot_model, name, 'prot')
