"""Minimal-edit preferences inspired by development error TYPES, not test text.

Unlike the first curriculum's easy wrong answers, each rejected response keeps
useful surrounding prose and changes the actual clinical action or arithmetic.
No held-out32 prompt, reference or output is imported.
"""
from __future__ import annotations
import copy
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from medft.config import load_config,write_json,file_hash,utc_now
from medft.data import read_jsonl,write_jsonl,normalize
from scripts.build_consultation_preferences import publish


def hard_pairs():
    rows=[]
    def add(key,group,question,good,bad,source,reason):
        rows.append(dict(id='v51-hard-'+key,case_id='hard-'+group,state='minimal_edit',task='consultation',
            source=source,origin='authored_minimal_edit_preference',review_status='engineering_review_not_clinician',
            prompt=[{'role':'user','content':question}],chosen=good,rejected=bad,rationale=reason,
            group_unit='error_family',negative_is_actual_model_output=False))
    critical=[
        '化验室来电说结果达到危急值，我现在心悸、浑身没力，但没听清指标。需要先查清全部项目再决定是否去医院吗？',
        '接到检验人员紧急通知，人发软、心跳乱，能等夜班上完明天再去取结果吗？',
        '家人的检验结果被通知要紧急处置，现在头晕心慌。网上能先给点建议，等周末再面诊吗？',
        '检验机构刚说我有危急值，本人明显乏力心慌；我想先多喝水休息两天。这样稳妥吗？',
        '项目名称没记住，但电话明确说是危急结果，现在我心慌、站着发软。没有完整报告是否只能先等？',
        '医院打电话让我立即处理异常检验结果，现在我也很不舒服，能只等网上医生回复再行动吗？',
    ]
    for i,q in enumerate(critical):
        common='危急值通知合并当前明显心慌乏力，可能需要紧急处理；没有记住项目不能降低紧急程度。'
        good=common+'现在联系通知你的医疗机构并立即到急诊，明显虚弱或无法安全到院时拨打120，请人陪同，不自行驾车。不要等取全报告、线上回复或休息后再决定，也不靠多喝水替代处理。能安全取得时带报告和用药清单，按医务人员指引行动。'
        bad=common+'先联系网上医疗平台获取初步意见。至于能否等到明天，要看具体项目；暂时保持休息、充分喝水，等查清全部指标后再决定要不要到医院。'
        add(f'critical-{i}','critical',q,good,bad,'authored_current_urgent_context','No conditional delay after symptomatic critical-value notification; no speculative fluid treatment.')
    eyes=[
        '半小时前左眼视野像被拉下一片灰帘子，没有疼痛，是否先休息等明天眼科开门？',
        '单眼突然缺了一块视野，像布挡着，看不清，非得严重到眼痛才去急诊吗？',
        '忽然一只眼闪光，新的飞蚊很多，还出现一道幕布样黑影，现在该做什么？',
        '眼睛突然看到黑幕，不是普通干涩。我想先睡一觉，等更看不见再去医院，可以吗？',
    ]
    for i,q in enumerate(eyes):
        common='突发单眼幕布样视野缺损，即使不疼，也可能涉及需要及时处理的视网膜等问题，文字不能确诊。'
        add(f'eye-{i}','eye',q,
            common+'现在到眼科急诊或就近急诊，不等睡醒、明天门诊或进一步加重。请人陪同，不自行驾车，记录起始时间并带相关病史。',
            common+'先尽快约一次普通眼科检查。如果以后疼痛、视力下降得更严重，再考虑立即去急诊；今晚可以先休息观察。',
            'https://www.nhs.uk/conditions/detached-retina-retinal-detachment/','Urgent referral must not be conditional on further deterioration or pain.')
    breath=[
        '家人静坐时还在大口喘，嘴唇发青，应该躺平休息等一等吗？',
        '没有运动也喘得只能说几个字，脸色发青，先开车送去门诊可以吗？',
        '气喘突然加重，现在坐着也难受、嘴唇紫，该先在家做什么？',
        '母亲已经喘得不能完整说话，又有口唇发紫，想强行让她平躺省力，对吗？',
    ]
    for i,q in enumerate(breath):
        common='静息呼吸困难、发青或不能完整说话是危险表现。'
        add(f'breath-{i}','breath',q,
            common+'立即拨打120，停止活动并请人陪伴。让本人保持呼吸较舒适的姿势，通常坐起较舒适，不强迫平躺；按接线指引行动，不自行驾车。不要等网上追问完成。',
            common+'先让患者强行平躺以减轻呼吸困难，尽量休息。如果过一会没有缓解，再联系医院安排普通检查。',
            'https://www.nhs.uk/conditions/shortness-of-breath/','Do not replace emergency action by forced flat positioning or waiting.')
    for i,q in enumerate(['刚才家人脸突然歪了，一只胳膊抬不了，时间记不准确，急救电话中应该说什么？','说话突然不清楚、左臂没力，后来稍微好点了，还需要叫急救吗，如何记录时间？']):
        common='突发面部、肢体或语言变化要按卒中等急症处理，不能在线排除。'
        add(f'stroke-{i}','stroke',q,common+'现在拨打120，即使好转也不取消求助。记录起病时间；不清楚就说明最后完全正常的时间，不猜时间。不自行驾车，不自行喂药或食物。',common+'先等症状自己恢复，记不清时间就随便估一个。等明天普通门诊再查，既然好转就无需急救。','https://www.nhs.uk/conditions/stroke/symptoms/','Current emergency action plus last-known-well time.')
    for i,q in enumerate(['我担心今晚会伤害自己，一个人在家，危险物品就在旁边，先独自静一静好吗？','我不想一个人待着，怕控制不住碰身边的危险物品，应该如何求助？']):
        common='你描述的是眼前的安全风险，需要现实中的支持。'
        add(f'crisis-{i}','crisis',q,common+'现在联系可信任的人到身边陪伴，不要独处；在能安全做到时远离危险物品，不争抢。可能马上伤害自己或他人时立即拨打110或120，告诉接线人员位置和风险。线上聊天不能替代这些行动。',common+'尽量不要麻烦别人，先一个人待着深呼吸、数数，等自己冷静了再决定要不要联系支持。','https://www.nhs.uk/mental-health/get-urgent-help-for-mental-health/','Companionship cannot be replaced by solitary calming exercises.')
    for i,q in enumerate(['通知单上超声部位不清楚，我有糖尿病，是否为了保险先不吃饭、不喝水、停药？','朋友做超声被要求空腹，我自己的项目还没确认，也应先按最严格准备吧？']):
        common='不同部位和项目的超声准备要求不同，不能照搬他人的安排。'
        add(f'prep-{i}','prep',q,common+'先联系预约科室核实完整项目、部位及空腹饮水要求，说明基础病和用药。未确认时不能一概建议空腹，更不要自行停药；按针对本人项目的通知执行。',common+'如果拿不准，最安全的是一律空腹、禁水并自行停药，按最严格标准做就不会出错。','https://www.nhs.uk/tests-and-treatments/ultrasound-scan/','Strictest preparation is not automatically safest, especially diabetes and medications.')
    for i,q in enumerate(['医院说eGFR由肌酐公式估算，这两项都正常能算两项相互独立的验证吗？','一项检测肌酐，另一项直接拿肌酐和年龄计算，能说互相独立所以排除所有肾病吗？']):
        common='如果该eGFR是用肌酐及年龄等资料计算的，两者有直接的数据依赖。'
        add(f'kidney-{i}','kidney',q,common+'它们不能算两份相互独立的测量证据；结合尿白蛋白、病史和持续变化解释，也不能据两项正常排除所有肾脏问题。',common+'尽管存在计算依赖，它们依然互相独立，两项正常就能证明所有肾脏问题都不存在。','https://www.niddk.nih.gov/health-information/kidney-disease/chronic-kidney-disease-ckd/tests-diagnosis','A derived result is not an independent replication of its input measurement.')
    for i,q in enumerate(['头晕时没有量血压，伴不伴胸痛也记不清，没说的情况就当没有吗？','我只是说这几天偶尔晕，病史还没整理，能直接确定不急、让我做颈椎磁共振吗？']):
        add(f'dizzy-{i}','dizzy',q,'没有提供的信息应保留为未知，不能当成已经否认。先了解头晕是旋转感、要晕倒还是不稳，每次多久、何时开始、与体位活动的关系；补充病史用药，有无胸痛、晕厥、突发无力或言语异常。出现这些危险表现时立即求助；没有这些情况也要按实际病程安排面诊，不能在信息不足时直接指定影像项目。','你没提到的胸痛、晕厥和神经症状都视为没有，所以肯定不急。直接安排颈椎磁共振就能解释所有头晕，不必补充病史。','https://www.nhs.uk/conditions/dizziness/','Unknown is not denied; neither diagnostic certainty nor routine neck imaging follows from an incomplete history.')
    # These are arithmetic reading problems, not invented universal medical cutoffs.
    for i,(name,value,unit,lo,hi) in enumerate([
        ('肌酐',2.4,'mg/dL',0.6,1.2),('肌酐',3.1,'mg/dL',0.7,1.3),
        ('肌酐',0.4,'mg/dL',0.6,1.2),('肌酐',84,'μmol/L',60,110),
        ('肌酐',138,'μmol/L',50,100),('某项目',12,'任意单位',4,9),
        ('某项目',4,'任意单位',4,9),('某项目',9,'任意单位',4,9),
        ('某项目',3,'任意单位',4,9),('某项目',7,'任意单位',4,9),
        ('某项目',130,'任意单位',135,145),('某项目',148,'任意单位',135,145)]):
        state='低于' if value<lo else '高于' if value>hi else '位于'
        badstate='低于' if state=='高于' else '高于'
        q=f'模拟读数练习，不作诊断：{name}{value} {unit}，本题给定范围{lo}至{hi} {unit}，包含两端。只判断数字位置并说明局限。'
        tail='这里只按题目给定区间读数，不说明病因、临床严重程度或用药；真实报告还需核对检测条件、症状和完整病史。'
        add(f'number-{i}','number',q,f'{value} {state}本题给定的{lo}至{hi}范围。'+tail,
            f'{value} {badstate}本题给定的{lo}至{hi}范围。'+tail,
            'exact_arithmetic_fictional_interval','The only changed fact is the interval relation; verify it arithmetically.')
    return rows


def main():
    import yaml
    dest=ROOT/'data/generated/v5-1-consultation-hard'
    if dest.exists():raise FileExistsError(dest)
    train=read_jsonl(ROOT/'data/generated/v5-consultation/train.jsonl')+hard_pairs()
    # Eye-action corrections introduce the previous retinal source family into
    # training. Remove ALL six retinal branches from validation, not just the
    # similar wording. The remaining three source groups stay fixed.
    val=[r for r in read_jsonl(ROOT/'data/generated/v5-consultation/validation.jsonl') if r['case_id']!='retinal']
    blocked={normalize(m['content']) for p in ROOT.glob('assets/*frozen*.jsonl') for r in read_jsonl(p) for m in r.get('messages',[]) if m['role']=='user'}
    assert all(normalize(m['content']) not in blocked for r in train for m in r['prompt'] if m['role']=='user')
    config=copy.deepcopy(load_config('configs/v5_consultation.yaml'))
    config['project']['name']='medft-v5-1-consultation-hard'
    config['data'].update(source='data/generated/v5-1-consultation-hard/corpus.jsonl',output_dir='data/generated/v5-1-consultation-hard')
    config['training']['output_dir']='outputs/v5-1-consultation-hard'
    for split,rows in [('train',train),('validation',val)]:publish(f'data/generated/v5-1-consultation-hard/{split}.jsonl',rows)
    corpus=[dict(id=r['id'],case_id=r['case_id'],task=r['task'],source=r['source'],messages=[{'role':'system','content':config['system_prompt']}]+r['prompt']+[{'role':'assistant','content':r['chosen']}]) for r in train+val]
    publish('data/generated/v5-1-consultation-hard/corpus.jsonl',corpus)
    cfg=ROOT/'configs/v5_1_consultation_hard.yaml'
    assert not cfg.exists();cfg.write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
    prev=__import__('json').loads((ROOT/'outputs/consultation-v5/protocol.json').read_text(encoding='utf-8'))
    write_json(ROOT/'outputs/consultation-v5-1/protocol.json',dict(created_at=utc_now(),parent_protocol_sha256=file_hash(ROOT/'outputs/consultation-v5/protocol.json'),
        train_rows=len(train),validation_rows=len(val),new_minimal_edit_pairs=len(hard_pairs()),
        added_safety_types=['critical_value_urgency','retinal_urgency','dyspnoea_position','unknown_vs_negative','unit_interval_direction','preparation_specificity'],
        objective='same reference-cached DPO plus chosen SFT, new run from base',
        selection='Minimum fixed18 validation loss in three unchanged source groups; six retinal validation branches removed because that source family now trains. Content gate on original36. Full strength then optional predeclared .5 only.32 condition holdout remains untouched and not used for selection.',
        data_limits='Synthetic minimum edits, not actual model outputs or physician annotations. Inspired by previously used development error types; original36 no longer independent.',
        hashes={str(p.relative_to(ROOT)):file_hash(p) for p in [cfg,dest/'train.jsonl',dest/'validation.jsonl',dest/'corpus.jsonl',ROOT/'assets/consultation_v5_holdout_frozen.jsonl']},
        protected=prev['protected']))
    print('Prepared',len(train),'train pairs,',len(val),'validation; additional',len(hard_pairs()))


if __name__=='__main__':main()
