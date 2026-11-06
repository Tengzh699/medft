"""Reduce stock boilerplate and make clinical replies specific to the history.

Engineering candidates, not clinician labels. The inspected evaluations remain
development checks. No evaluation text or output is imported as an example.
"""
from __future__ import annotations
import copy
import json
import random
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from medft.config import load_config, project_path, write_json, file_hash, utc_now
from medft.data import build_data, read_jsonl, normalize
from medft.training import plan_training
from scripts.build_complete_data import authored_rows, numerical_rows, pair, fmt, group_complete_answers, safe_publish, SCENARIOS

# Distinct follow-up questions and current actions, rather than one shared
# symptom-agnostic paragraph. No drug doses or definitive diagnoses.
CLINICAL = {
 'breath': ('什么时候开始气喘，活动和休息时各怎样？有无胸痛、发青、晕厥，既往心肺疾病和用药是什么？',
            '可以记录活动种类、持续时间、气喘程度及休息后的恢复情况，避免硬撑。若反复发生或影响活动，到呼吸内科或全科评估。',
            '静息时也呼吸困难、口唇发青且讲话费劲，需现在拨打120。停止活动，请人陪伴并按接线人员指导行动，不自行驾车。说明发生时间及心肺疾病、用药信息，不能等后天预约。'),
 'neuro': ('麻木是哪一侧、什么部位，每次多久？与姿势有关吗，是否突然无力、脸歪或说话含糊？既往病史和用药也请说明。',
           '可记录姿势、麻木部位和持续时间，不要长期压迫同一部位。持续或反复时到神经内科或全科面诊。',
           '突发脸歪、言语含糊、一侧胳膊抬不起来，需要立即拨打120，不等恢复。记下开始时间；不确定时说明最后一次完全正常的时间。请人陪伴，不自行驾车，不自行喂药或食物。'),
 'abdomen': ('腹部具体哪里不舒服，从什么时候开始，怎样变化？是否发热、反复呕吐、出血、不能进食或走动，是否可能怀孕？',
             '可以记录疼痛位置、进食和排便的关系，保持能耐受的日常饮食，不自行决定抗生素或检查套餐。持续不缓解时面诊消化内科或全科。',
             '腹痛明显加重、发热且走动也痛，需要立即到急诊评估，不能继续等预约。请人陪同；疼痛剧烈、晕厥或不能安全到达时拨打120。带已知病史和用药，不能在线确定病因或等追问结束才出发。'),
 'bleeding': ('不适的位置、持续时间及与进食关系是什么？有没有呕血、黑便、晕厥，是否服用抗凝药或其他药？',
              '可记录进食后不适的时间和诱因，持续或反复时到消化内科面诊。不要自行决定停药、换药或胃镜项目。',
              '已经呕血且腹痛加重，应立即联系120或急诊，不能等常规预约或先在家观察。请人陪伴，不自行驾车，说明出血和发病时间，带上用药信息，按急救人员指导行动。'),
 'vision': ('是一只眼还是两只眼，从何时开始？是否突然视力下降、出现闪光、飞蚊或像帘子遮挡，有无眼痛或外伤？',
            '可减少连续近距离用眼，改善照明并安排休息；若疲劳持续不缓解，可到眼科检查。不要凭疲劳自行使用治疗性眼药水。',
            '突然一眼出现幕布样视野遮挡，需要现在到眼科急诊或就近急诊，不能先睡一觉或等加重。请人陪同，不自行驾车，记录开始时间；无法安全到达时联系120。'),
 'injury': ('何时、怎样撞到头，是否失去意识、持续头痛或呕吐？有没有抗凝用药、出血疾病或新的无力、嗜睡？',
            '按实际情况继续留意头痛、呕吐、精神和功能变化；对受伤程度不确定时联系医疗人员评估。不要仅凭在线描述保证安全。',
            '抗凝用药背景下头部受伤并反复呕吐，应立即到急诊或拨打120，不能等呕吐停止。请人陪伴，说明受伤时间、药名和最后用药时间，按急救人员指导行动，不自行调整抗凝药。'),
 'pregnancy': ('大约孕几周，是否有腹痛、阴道出血、晕厥或肩部疼痛？既往妊娠和当前产检情况怎样？',
               '目前是在讨论就诊安排，可联系产科确认产检时间，整理孕周、既往检查和用药，不自行停药或补充治疗。',
               '孕早期突然剧烈腹痛并阴道出血，需要立即到急诊或妇产科急诊，不等网上回复。请人陪同；晕厥、明显虚弱、出血多或不能安全到院时拨打120。带孕周、检查和用药信息。'),
 'allergy': ('接触什么产品、何时开始、范围和变化怎样？有没有面唇舌肿胀、呼吸困难、头晕，是否有既往反应或新用药？',
             '先避免继续接触可疑洗剂，记录接触与皮肤变化，温和防护，避免混用刺激性用品；持续或反复时看皮肤科。',
             '面舌肿胀、呼吸困难并头晕，需要立即拨打120。不要继续接触可疑物，请人陪伴，按接线人员指导处理，不等待皮肤症状自行消退。'),
 'chest': ('体力变化何时开始，与运动或休息有什么关系？有无胸痛、冷汗、气短或晕厥，是否有心血管病史和用药？',
           '可从能恢复的活动量开始，记录时间、强度和恢复情况，逐步调整；反复明显气短或体力下降时面诊评估。',
           '持续压榨样胸痛伴冷汗和气短，应立即拨打120，不自行驾车或先去完成原预约。停止活动，请人陪伴，说明开始时间和用药，按急救人员指导行动。'),
 'poison': ('目前是否确实发生误服？若有，何时、可能是什么、多少量，现在意识和呼吸怎样？不要为了查清所有信息延误求助。',
            '目前没有误服，应把药物保留在原包装，按要求储存并置于儿童拿不到的位置，不用饮料瓶分装，不混装不明药片。',
            '孩子误服不明药片且犯困，应立即联系120或急诊。不自行催吐，不喂食喂水，不让孩子独处；能安全取得时带包装和剩余药片，不为辨认药物延误救助。'),
 'mental': ('最近心情变化持续多久，对睡眠和生活有什么影响？现在是否有伤害自己或他人的想法、计划或可取得的危险物品，身边有人支持吗？',
            '目前没有伤害打算且有人陪伴，可联系可信任的人，讨论心理咨询或精神心理科的评估安排。可以分享愿意说的感受，不必独自承担。',
            '现在先联系可信任的人到身边陪伴，不要独处，远离危险物品和地点，不争抢物品。担心马上伤害自己或他人时，立即拨打110或120，说明位置和当前风险，按接线人员指导行动。'),
 'critical': ('请保留完整项目、数值、单位、参考范围和日期；有没有检验科的危急值通知，现在有无明显不适？',
              '目前没有不适或危急通知，可以按日期整理完整报告，核对单位和检测条件，向开单医生解释疑问，不靠箭头自行诊断。',
              '已有危急值通知并心慌乏力，应现在联系检验机构和急诊，必要时拨打120，不能等补齐项目再行动。请人陪伴、带报告与用药信息，不自行驾车。'),
}

EXTRA = [
 ('symptom_collection', '头痛说不清位置，想在看诊前整理一下，应该怎么讲？', '先记下从何时开始、每次多久、疼痛部位和性质，以及是否逐渐加重。补充发热、呕吐、外伤、用药和既往类似发作；不知道的内容如实说明。若突然出现从未有过的剧烈头痛，或伴无力、说话不清、意识异常，应立即就医，不能先等记录完善。'),
 ('symptom_collection', '嗓子不舒服两天，我还没量体温，接下来告诉你哪些信息合适？', '两天的病程已知道，不必重新猜开始时间；没有量体温不等于没有发热。请补充疼痛与吞咽、饮水的关系，有无咳嗽、呼吸困难、声音变化，以及接触史和用药。若不能吞咽口水、呼吸费劲或迅速加重，应及时急诊评估；其他情况可带记录到门诊判断。'),
 ('examination', '膝盖不舒服，X线与磁共振各看什么，哪种贵就更适合吗？', '价格不能决定适合哪种检查。X线有助于观察骨性结构和关节间隙；磁共振提供软骨、韧带等软组织信息，但并非每种膝痛都需要。医生先结合受伤经过、疼痛位置、负重和查体明确问题，再考虑是否检查及选择项目。带上旧影像和植入物信息，询问结果是否会改变后续安排。'),
 ('triage', '最近几次鼻出血，已经停了，不知道先看哪个门诊。', '反复鼻出血可先到耳鼻喉科或全科评估，说明频率、每次多久、哪侧出血，以及用药、外伤和其他部位出血情况。已经止住不代表能凭文字确定原因。若再次出血量多、持续不能控制，或伴晕厥、明显乏力和呼吸困难，应及时急诊；不要仅等常规预约。'),
 ('triage', '指甲颜色最近慢慢变了，没有给你照片，能先咨询哪里？', '可以先看皮肤科，带上变化前后的照片，说明哪几个指甲、多久、外伤或染色接触及用药。指甲变化可能有多种原因，单凭颜色不能确诊，也不宜自行开始抗真菌药。若有明显疼痛、红肿、伤口或迅速恶化，应提前就医。'),
 ('symptom_collection', '肩膀活动不太方便，病历上的病程栏我还空着。', '请按实际情况填写何时开始、哪侧肩、哪些动作受限、是否外伤，以及睡眠和日常活动受影响的程度；记不清就说明大概时间。没有提供的信息不能当作已经没有。若伴胸部不适、气短、冷汗，或突然肢体无力等，应及时就医；一般持续活动受限可安排骨科、康复科或全科面诊。'),
 ('examination', '肺功能检查一次没异常，为什么医生仍要听我描述运动时的不适？', '检查反映特定时间和方法下的结果，可能没有重现运动或间歇发作时的情况。医生还要了解诱因、持续时间、恢复过程、其他症状及病史，判断是否需要进一步评估。一次正常既不能保证以后都没问题，也不表示一定有病。可记录发作时的活动和变化，出现严重呼吸困难时及时求助。'),
 ('examination', '最近总觉得累，检查套餐宣传能一次排除所有疾病，我该怎样问？', '先说明疲劳多久、睡眠饮食、活动变化、用药和伴随症状，再问每项检查要回答什么问题、结果会怎样影响处理，以及风险和替代选择。没有一种套餐能一次排除所有疾病，项目多还可能带来偶然异常和后续检查。应根据个人情况选择，并问清何时复评及哪些变化需要提前联系医生。'),
 ('knowledge_qa', '成年孩子准备做心理咨询，家长能提供哪些实际帮助？', '先问本人希望得到什么帮助，例如预约、交通或在外等候；进入咨询室及信息分享由本人和咨询师商量。尊重隐私，不擅自索取谈话记录，不替他回答。平时可以倾听和陪伴，协助他愿意安排的日常事务。若出现眼前的伤害危险，则优先联系现实支持和紧急服务。'),
 ('knowledge_qa', '骑车的量想慢慢加上去，日记里除了路程写什么？', '记下时间、路线或坡度、主观强度，以及骑行中和之后的疲劳、疼痛、气促和恢复情况。第二天能否正常活动也是反馈。逐步改变一个因素，给身体恢复时间，不只追求里程。出现胸痛、晕厥或明显呼吸困难应停止活动并及时就医；基础病或长期未运动时先讨论合适范围。'),
 ('examination', '检查通知里的部位看不清，我想按朋友做另一种超声的准备照做。', '先向预约科室核实完整项目、部位和具体准备。不同超声的要求可能不同，别人的空腹、饮水或用药安排不一定适用；通知不清楚也不能推断为必须或无需准备。把自己的用药和相关基础病说明，按检查机构针对该项目的要求执行。'),
 ('knowledge_qa', '血清肌酐与报告上根据它算出的肾功能估值，有什么联系？', '肌酐是血液测量指标，受肾脏清除、肌肉量、饮食和部分药物等影响。肌酐相关公式用它及年龄、性别等信息估算eGFR，因此两者不是相互独立的验证；eGFR主要估算肾小球滤过功能。还应结合尿检、病史及持续变化，不能只凭参考范围内的肌酐排除全部肾脏问题。'),
 ('knowledge_qa', '一张血糖采血结果和糖化血红蛋白，观察时间一样吗？', '不一样。空腹或随机血糖都是采样时点的测量，解释时要看进食时间和采样条件；空腹血糖不是过去一天或几天的平均值。糖化血红蛋白通常反映近两到三个月平均血糖，红细胞寿命变化、输血和部分贫血等会影响解释。两种结果可互相补充，不能只凭一项自行调整治疗。'),
 ('refusal', '疫苗登记要填过敏，我忘了以前的反应，能编一个标准答案吗？', '不能编造过敏史。不记得就如实说明未知，尽量查找既往记录，描述记得的产品、时间、表现和是否接受处理。药品不适与确诊过敏并不完全相同，由接种或医疗人员核实并评估安排。可以帮你整理真实信息清单，不替你捏造答案或决定能否接种。'),
 ('refusal', '同事没同意，我把他的完整报告发到工作群让大家分析可不可以？', '不应未经同意公开传播同事的报告，其中可能包含身份和敏感健康信息。先征求本人意愿，通过适当医疗渠道提供必要内容；简单遮住姓名也不保证无法识别个人。若只是讨论知识，可以使用抽象问题或有许可的教学材料，避免扩散真实个人资料。'),
 ('knowledge_qa', '饮料每100毫升标47千焦，我喝150毫升，这个数就是一天的能量吗？', '这次饮料的能量为47×150÷100＝70.5千焦，单位沿用标签。它只包含本次150毫升饮料，不是全天摄入，也不是每日需要量；应把当天其他食物和饮料加在一起，再结合个人情况讨论饮食。千焦与千卡是不同能量单位，不能把70.5直接写成千卡。'),
]


def natural_authored():
    rows = authored_rows()
    scenarios = {f'complete-clinical-{s[0]}': s for s in SCENARIOS}
    for row in rows:
        if not row.get('clinical_state'): continue
        key, dept, opening, stable, danger, summary, url = scenarios[row['case_id']]
        ask, follow, urgent = CLINICAL[key]
        state = row['clinical_state']
        if len(row['messages']) > 2:
            row['messages'][1]['content'] = ask
        if state in {'direct', 'change'}:
            answer = urgent
        elif state == 'hypothetical':
            answer = f'你明确说现在是{stable}，危险情形是将来的假设，不把它当作当前病史。{follow}若将来真的出现所述危险表现，届时的做法是：{urgent}'
        elif state == 'correction':
            answer = f'已按你最新说明更正：{stable}；之前转述的危险表现不算现在发生。{follow}若后来出现{summary}，应及时急救处理。'
        else:
            answer = f'你说明{stable}。{follow}若后来出现{summary}，需立即联系急救或急诊；此前较轻不能排除后续变化。'
        row['messages'][-1]['content'] = answer
    # One source group per report/composition is retained; variants do not
    # become fictitious independent clinical cases.
    numeric = numerical_rows()
    seen = Counter()
    for row in numeric:
        family = row['case_id']
        limit = 1 if row.get('arithmetic') else 3 if row['id'].startswith('complete-glucose') else 1
        if seen[family] >= limit: continue
        seen[family] += 1
        if 'calculation' in row:
            data = row['calculation']
            answer = row['messages'][-1]['content'].split('如要了解全天摄入')[0]
            row['messages'][-1]['content'] = answer + '全天摄入还需合计其他食物和饮料；是否合适要结合个人需要。'
            assert Decimal(data['per_100']) * Decimal(data['portion']) / 100 == Decimal(data['result'])
        elif 'arithmetic' in row:
            answer = row['messages'][-1]['content'].split('仍需结合症状')[0]
            row['messages'][-1]['content'] = answer + '这里仅做数值阅读，不据此确定病因。机构区间为教学模拟；带完整报告、症状和病史向医生核实。'
        rows.append(row)
    for i, (task, question, answer) in enumerate(EXTRA):
        rows.append(pair(f'natural-extra-{i:02}', f'natural-extra-{i:02}', task, question, answer))
    for row in rows:
        row['curriculum_version'] = 'specific-answer-v1'
        row['source'] = 'author_created_specific_answer_curriculum'
        row['origin'] = 'authored_specific'
        row.pop('answer_group', None)
    return rows


def link_known_topics(rows):
    """Conservatively keep these explicitly related paraphrases together."""
    groups = {
        'snapshot': ['complete-qa-001', 'complete-qa-011', 'natural-extra-06'],
        'packages': ['complete-qa-009', 'natural-extra-07'],
        'consultation-support': ['complete-qa-005', 'complete-qa-006', 'natural-extra-08'],
        'activity': ['complete-qa-008', 'complete-qa-022', 'natural-extra-09'],
        'ultrasound': ['complete-qa-010', 'natural-extra-10'],
        'kidney': ['complete-qa-016', 'natural-extra-11'],
        'glucose-time': ['complete-qa-014', 'complete-qa-015', 'natural-extra-12'],
        'truthful-history': ['complete-refusal-001', 'natural-extra-13'],
        'medical-privacy': ['complete-refusal-002', 'complete-refusal-007', 'natural-extra-14'],
    }
    aliases = {ident: f'natural-topic-{topic}' for topic, identifiers in groups.items() for ident in identifiers}
    for row in rows:
        canonical = aliases.get(row['id'])
        if row['id'] == 'complete-refusal-005': canonical = 'complete-clinical-poison'
        if row['id'] == 'complete-refusal-006': canonical = 'complete-clinical-mental'
        if canonical:
            row['original_case_id'] = row['case_id']
            row['case_id'] = canonical
    return rows


def main():
    # The earlier unlinked draft remains in v3-4-natural, before any training.
    variant = 'v3-4-natural-grouped'
    config = copy.deepcopy(load_config('configs/v3_3_attention_control.yaml'))
    config['project']['name'] = 'medft-' + variant
    config['data'].update(source=f'data/generated/{variant}/corpus.jsonl', output_dir=f'data/processed/{variant}')
    config['training']['output_dir'] = f'outputs/{variant}'
    replay = read_jsonl(project_path('data/generated/v3-1-complete/base_replay.jsonl'))
    rows = group_complete_answers(link_known_topics(natural_authored() + replay))
    forbidden = {normalize(m['content']) for name in ['evaluation_v1.jsonl', 'evaluation_v2.jsonl', 'evaluation_complete_frozen.jsonl', 'evaluation_retention_probes.jsonl']
                 for case in read_jsonl(project_path('assets') / name) for m in case['messages'] if m['role'] == 'user'}
    assert not any(normalize(m['content']) in forbidden for row in rows for m in row['messages'] if m['role'] == 'user')
    random.Random(42).shuffle(rows)
    safe_publish(project_path(config['data']['source']), rows)
    cfgpath = project_path('configs/v3_4_natural_grouped.yaml')
    if cfgpath.exists(): assert load_config(cfgpath) == config
    else: cfgpath.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
    evalconfig = copy.deepcopy(config)
    evalconfig['data']['evaluation_source'] = 'assets/evaluation_complete_frozen.jsonl'
    evalconfig['data']['evaluation_exclusions'] = ['assets/evaluation_v1.jsonl', 'assets/evaluation_v2.jsonl']
    evalpath = project_path('configs/v3_4_natural_grouped_eval.yaml')
    if evalpath.exists(): assert load_config(evalpath) == evalconfig
    else: evalpath.write_text(yaml.safe_dump(evalconfig, allow_unicode=True, sort_keys=False), encoding='utf-8')
    stats = build_data(config)
    preparation = {'created_at': utc_now(), 'source_sha256': file_hash(project_path(config['data']['source'])),
                   'rows_before_cleaning': len(rows), 'by_task': dict(Counter(r['task'] for r in rows)),
                   'origins': dict(Counter(r['origin'] for r in rows)),
                   'rationale': 'Tailor questions/actions to each clinical family; thin numerical variants to reduce repeated stock endings; remove drug/treatment boilerplate from food arithmetic; add 16 different health-literacy situations. Reuse the same screened base replay.',
                   'review_status': 'unreviewed',
                   'evaluation_role': 'All 36+29 evaluation outputs have been inspected. Revision responds to development error categories; related topics recur. No exact evaluation prompts or model outputs are added. This cannot establish blind generalization.',
                   'stats': stats, 'plan': plan_training(config)}
    write_json(project_path(f'data/generated/{variant}/preparation.json'), preparation)
    print(json.dumps(preparation, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
