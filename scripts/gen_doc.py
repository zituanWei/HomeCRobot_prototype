# -*- coding: utf-8 -*-
"""
生成《家庭陪护机器人项目文档》Word 文件
=====================================
内容数据在 doc_content.yaml，本脚本只负责排版。

这样分开的好处：改内容只需编辑 yaml，不用碰 Python。

用法:
    python scripts/gen_doc.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from docx import Document
from docx.shared import Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

OUT = ROOT / "项目文档.docx"


def safe_out():
    """选一个可写的输出路径

    如果 项目文档.docx 正被 Word 打开，直接写会 PermissionError。
    此时自动换一个文件名，而不是让整个生成过程失败。
    """
    if not OUT.exists():
        return OUT
    try:
        with open(OUT, 'ab'):
            return OUT
    except PermissionError:
        alt = ROOT / '项目文档_new.docx'
        print(f'  [提示] {OUT.name} 正被占用（可能已在 Word 中打开）')
        print(f'         本次输出到: {alt.name}')
        return alt
DATA = Path(__file__).parent / "doc_content.yaml"

ACCENT = RGBColor(0x1F, 0x4E, 0x79)
GREY = RGBColor(0x59, 0x59, 0x59)
RED = RGBColor(0xC0, 0x00, 0x00)
ORANGE = RGBColor(0xBF, 0x60, 0x00)

FONT = '微软雅黑'


def load_data():
    """读 yaml（不依赖第三方库，简易解析够用）"""
    try:
        import yaml
        return yaml.safe_load(DATA.read_text(encoding='utf-8'))
    except ImportError:
        pass
    # 没有 pyyaml 时退化为：直接 exec 成 dict 不可行，这里提示
    raise RuntimeError('需要 pyyaml：pip install pyyaml')


def setup(doc):
    st = doc.styles['Normal']
    st.font.name = FONT
    st.font.size = Pt(10.5)
    st.element.rPr.rFonts.set(qn('w:eastAsia'), FONT)
    for name, size in [('Heading 1', 16), ('Heading 2', 13), ('Heading 3', 11.5)]:
        s = doc.styles[name]
        s.font.name = FONT
        s.font.size = Pt(size)
        s.font.color.rgb = ACCENT
        s.font.bold = True
        s.element.rPr.rFonts.set(qn('w:eastAsia'), FONT)
    for sec in doc.sections:
        # 必须显式设为 A4：python-docx 默认是 Letter（21.59cm），
        # 按 Letter 排版的内容在 A4 下会横向溢出被裁掉
        sec.page_width = Cm(21.0)
        sec.page_height = Cm(29.7)
        sec.left_margin = sec.right_margin = Cm(2.0)
        sec.top_margin = sec.bottom_margin = Cm(2.0)


def mono(doc, text, size=8.5):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.name = 'Consolas'
    r.font.size = Pt(size)
    return p


def bullets(doc, items):
    for it in items:
        doc.add_paragraph(str(it), style='List Bullet')


def table(doc, spec):
    """插入表格

    注意两点，都是踩过坑的：
    1. **不要**手工设单元格宽度。按 Letter（21.59cm）算的列宽在 A4 下会撑出页面。
    2. 必须显式声明表格宽度与自动布局，否则某些样式会带入模板里的固定宽度，
       同样导致右侧溢出被裁。
    """
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn as _qn

    hdrs = spec['headers']
    t = doc.add_table(rows=1, cols=len(hdrs))
    t.style = 'Light Grid Accent 1'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True

    # 显式声明：表格宽度 = 100% 页面可用宽度，布局为自动
    tblPr = t._tbl.tblPr
    for tag, attrs in (
        ('w:tblW', {_qn('w:w'): '5000', _qn('w:type'): 'pct'}),
        ('w:tblLayout', {_qn('w:type'): 'autofit'}),
    ):
        el = tblPr.find(_qn(tag))
        if el is None:
            el = OxmlElement(tag)
            tblPr.append(el)
        for k, v in attrs.items():
            el.set(k, v)

    for i, h in enumerate(hdrs):
        c = t.rows[0].cells[i]
        c.text = ''
        r = c.paragraphs[0].add_run(str(h))
        r.bold = True
    for row in spec['rows']:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            if i < len(cells):
                cells[i].text = str(v)

    doc.add_paragraph()   # 表格后留间距
    return t


def warn(doc, text):
    p = doc.add_paragraph()
    r = p.add_run('⚠ ' + text)
    r.font.color.rgb = ORANGE
    r.italic = True


def code_block(doc, text, size=8.5):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.name = 'Consolas'
    r.font.size = Pt(size)


def build(d):
    doc = Document()
    setup(doc)

    # ------- 封面 -------
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run('家庭陪护机器人')
    r.bold = True
    r.font.size = Pt(28)
    r.font.color.rgb = ACCENT

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run('HomeCRobot')
    r.bold = True
    r.font.size = Pt(22)
    r.font.color.rgb = ACCENT

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(d['subtitle'])
    r.font.size = Pt(15)
    r.font.color.rgb = GREY

    # 代码规模
    files = total = 0
    for f in ROOT.rglob('*.py'):
        s = str(f)
        if '__pycache__' in s or 'models' in s:
            continue
        files += 1
        total += len(f.read_text(encoding='utf-8', errors='ignore').splitlines())

    doc.add_paragraph()
    for line in [
        '基于 Qwen2.5-VL 的本地多模态陪护系统',
        f'代码规模：{files} 个 Python 文件 / 约 {total:,} 行',
        '本文档由脚本生成，可随项目更新重新生成',
    ]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run(line).font.size = Pt(10.5)

    doc.add_page_break()

    # ------- 一、项目概述 -------
    doc.add_heading('一、项目概述', level=1)
    for para in d['overview']:
        if isinstance(para, dict):
            for k, v in para.items():
                if k == '段落':
                    doc.add_paragraph(v)
                elif k == '目标':
                    doc.add_paragraph('核心目标：')
                    bullets(doc, v)
        else:
            doc.add_paragraph(str(para))
    doc.add_heading('1.3 当前完成状态', level=2)
    table(doc, d['status'])

    # ------- 二、系统架构 -------
    doc.add_heading('二、系统架构', level=1)
    doc.add_heading('2.1 整体数据流', level=2)
    doc.add_paragraph('系统采用事件驱动架构，三路输入统一入队后由主循环分发：')
    code_block(doc, d['architecture_flow'])
    doc.add_paragraph(d['architecture_note'])

    doc.add_heading('2.2 模块划分', level=2)
    doc.add_heading('核心层 core/', level=3)
    table(doc, d['core_modules'])
    doc.add_heading('扩展层 modules/', level=3)
    table(doc, d['ext_modules'])
    doc.add_heading('工具层 utils/', level=3)
    table(doc, d['utils_modules'])

    doc.add_heading('2.3 并发与线程安全', level=2)
    doc.add_paragraph('系统有多个线程同时运行，共享资源必须保护，否则会出现难以复现的故障：')
    table(doc, d['concurrency'])

    # ------- 三、功能模块详解 -------
    doc.add_heading('三、功能模块详解', level=1)

    doc.add_heading('3.1 视觉对话', level=2)
    doc.add_paragraph(d['func_vision'])
    warn(doc, d['func_vision_warn'])

    doc.add_heading('3.2 人脸识别', level=2)
    doc.add_paragraph('存储结构（按人分目录，保留原图）：')
    code_block(doc, d['face_tree'], 8.5)
    doc.add_paragraph('关键技术点：')
    bullets(doc, d['face_points'])

    doc.add_heading('3.3 语音输入', level=2)
    doc.add_paragraph('处理链路：')
    code_block(doc, d['asr_chain'])
    warn(doc, d['asr_note'])

    doc.add_heading('3.4 语音输出', level=2)
    doc.add_paragraph('支持多引擎，按音质与延迟权衡选择：')
    table(doc, d['tts_engines'])
    doc.add_paragraph('自动降级机制，避免在线引擎慢导致长时间无声音：')
    bullets(doc, d['tts_fallback'])
    doc.add_paragraph(d['tts_preload'])

    doc.add_heading('3.5 主动关怀', level=2)
    doc.add_paragraph('采用双层检测，兼顾响应速度与算力开销：')
    table(doc, d['monitor_levels'])
    doc.add_paragraph('主动行为：')
    bullets(doc, d['monitor_actions'])

    doc.add_heading('3.6 引导注册', level=2)
    doc.add_paragraph('当用户说了身份相关的话但系统认不出时，主动引导注册：')
    code_block(doc, d['enroll_flow'], 9)
    warn(doc, d['enroll_warn'])

    # ------- 四、技术选型 -------
    doc.add_heading('四、技术选型', level=1)
    table(doc, d['tech_choices'])

    # ------- 五、实测数据 -------
    doc.add_heading('五、实测数据', level=1)
    doc.add_paragraph('以下数据为开发过程中在本机（RTX 4060 Laptop 8GB）实测所得，'
                      '用于说明系统特性与边界，不同硬件会有差异。')

    doc.add_heading('5.1 语音链路', level=2)
    table(doc, d['measured_audio'])

    doc.add_heading('5.2 视觉能力', level=2)
    table(doc, d['measured_vision'])

    doc.add_heading('5.3 输出稳定性', level=2)
    doc.add_paragraph('同一张图、同一个问题重复提问，观察答案一致性：')
    table(doc, d['measured_stability'])
    warn(doc, d['stability_warn'])

    doc.add_heading('5.4 陪护场景判断', level=2)
    table(doc, d['measured_scene'])
    warn(doc, d['scene_warn'])

    # ------- 六、目录结构 -------
    doc.add_heading('六、目录结构', level=1)
    code_block(doc, d['dir_tree'], 8.5)

    # ------- 七、脚本工具 -------
    doc.add_heading('七、脚本工具说明', level=1)
    table(doc, d['scripts_table'])

    # ------- 八、部署与使用 -------
    doc.add_heading('八、部署与使用', level=1)
    doc.add_heading('8.1 环境要求', level=2)
    bullets(doc, [
        'Python 3.10 及以上，实测 3.12',
        'NVIDIA 显卡，8GB 显存可运行 3B 四比特量化',
        '摄像头、麦克风、扬声器',
    ])
    doc.add_heading('8.2 启动流程', level=2)
    code_block(doc, d['deploy_cmd'], 9)
    doc.add_heading('8.3 常用命令', level=2)
    table(doc, d['commands'])
    doc.add_paragraph(d['nl_examples'])

    # ------- 九、已知限制 -------
    doc.add_heading('九、已知限制', level=1)
    doc.add_heading('9.1 模型能力边界', level=2)
    bullets(doc, d['limit_model'])
    doc.add_heading('9.2 架构限制', level=2)
    bullets(doc, d['limit_arch'])

    # ------- 十、后续维护 -------
    doc.add_heading('十、后续维护指引', level=1)
    doc.add_paragraph('本节供后续修改者参考。')

    doc.add_heading('10.1 配置在哪里改', level=2)
    doc.add_paragraph('所有可调参数集中在 config.py：')
    table(doc, d['config_map'])

    doc.add_heading('10.2 常见修改场景', level=2)
    table(doc, d['howto'])

    doc.add_heading('10.3 修改后的验证顺序', level=2)
    code_block(doc, d['verify_cmd'], 9)

    doc.add_heading('10.4 更新本文档', level=2)
    doc.add_paragraph(d['update_doc'])
    doc.add_paragraph('内容数据在 scripts/doc_content.yaml，'
                      '排版脚本在 scripts/gen_doc.py。改内容只需编辑前者。')

    doc.add_heading('10.5 排错入口', level=2)
    table(doc, d['troubleshoot'])

    out = safe_out()
    doc.save(str(out))
    return out


def main():
    if not DATA.exists():
        print(f'缺少内容文件: {DATA}')
        return 1
    d = load_data()
    out = build(d)
    print('=' * 60)
    print(' 项目文档已生成')
    print('=' * 60)
    print(f'  {out}')
    try:
        print(f'  大小 {out.stat().st_size / 1024:.1f} KB')
    except Exception:
        pass
    print()
    print('  含 10 个版块：项目概述 / 系统架构 / 功能详解 / 技术选型 /')
    print('  实测数据 / 目录结构 / 脚本工具 / 部署使用 / 已知限制 / 维护指引')
    print()
    print('  内容数据: scripts/doc_content.yaml（改内容编辑这个）')
    print('  排版脚本: scripts/gen_doc.py（改结构编辑这个）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
