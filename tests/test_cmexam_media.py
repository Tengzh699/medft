import unittest
from scripts.audit_cmexam_media import media_hints


class MissingMediaTests(unittest.TestCase):
    def test_explicit_unavailable_image_and_table(self):
        self.assertTrue(media_hints('头颅CT如图（暂无图），诊断是什么？'))
        self.assertTrue(media_hints('随访结果见下表，保护指数是多少？'))
        self.assertTrue(media_hints('化学结构如下图，代谢途径是什么？'))

    def test_textual_findings_are_not_missing_media(self):
        self.assertFalse(media_hints('心电图示ST段下降。超声心动图示左房60mm。'))
        self.assertFalse(media_hints('镜下表现为炎性肉芽组织，叶下表面灰绿色。'))
        self.assertFalse(media_hints('红外光谱图中1650cm⁻¹处有强吸收峰。'))


if __name__ == '__main__':
    unittest.main()
