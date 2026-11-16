"""Independent inventory/selection/gate verification and concise experiment report."""
import json
import statistics
from pathlib import Path

import torch

from research.baseline.common import ROOT, sha256_file
from research.diagnostics.p2_config_check import OUT, DOC, SEEDS, CHECKS, save, make_model, matrices, loss


def passed(m):
    return m['response']<=.05 and m['H15_mse']<=.25*m['ignore_input_H15_mse']


def verify():
    snapshot=json.loads((OUT/'source_snapshot.json').read_text())
    assert snapshot['protocol_sha256']==sha256_file(OUT/'protocol_before_run.md')
    for name,expected in snapshot['files'].items():assert sha256_file(ROOT/name)==expected,name
    environment=json.loads((OUT/'environment.json').read_text())
    for name,expected in environment['extra_sources'].items():assert sha256_file(ROOT/name)==expected,name
    runs={p.parent.name:json.loads(p.read_text()) for p in (OUT/'runs').glob('*/result.json')}
    expected={'tiny-lr0.001'}
    tiny=runs['tiny-lr0.001']
    if tiny['status']!='complete' or not passed(tiny['final_validation_or_tiny_train']):expected.add('tiny-lr0.01')
    tuning=[];candidates=[]
    for noenc in (False,True):
        for lr in (1e-4,1e-3,1e-2):
            for scheduler in ('constant','epochmean'):
                name=f'tune-lr{lr}-{scheduler}-noenc{int(noenc)}'
                expected.add(name);tuning.append(name);r=runs[name]
                assert r['cfg']=={'lr':lr,'scheduler':scheduler,'epochs':1000,'noenc':noenc}
                if r['status']=='complete' and passed(r['final_validation_or_tiny_train']):candidates.append(r)
        if candidates:break
    selected=json.loads((OUT/'selected_config.json').read_text())
    gates=json.loads((OUT/'stage_gates.json').read_text())
    assert selected['tuning_runs']==tuning
    if candidates:
        best=min(candidates,key=lambda r:(r['updates'],r['final_validation_or_tiny_train']['response'],r['name']))
        assert selected['status']=='SELECTED' and selected['name']==best['name']
        assert selected['cfg']=={**best['cfg'],'epochs':best['epochs_completed']}
        confirmed=[];general=[]
        for seed in SEEDS:
            name=f'confirm-compatible-s{seed}';expected.add(name);confirmed.append(name)
            for geometry in ('SPD','general'):
                name=f'original-a0-{geometry}-s{seed}';expected.add(name)
                if geometry=='general':general.append(name)
        assert set(gates['confirmation'])==set(confirmed)
        assert set(gates['original_comparison'])=={f'original-a0-{geometry}-s{seed}' for geometry in ('SPD','general') for seed in SEEDS}
        compat_ok=all(runs[n]['status']=='complete' and passed(runs[n]['final_validation_or_tiny_train']) for n in confirmed)
        general_ok=all(runs[n]['status']=='complete' and passed(runs[n]['final_validation_or_tiny_train']) for n in general)
        assert gates['compatible_all_five_pass']==compat_ok
        assert gates['original_general_all_five_pass']==general_ok
        stress=[]
        if compat_ok and general_ok:
            stress=[f'stress-a{a}-general-s{seed}' for a in (.5,1.,2.) for seed in SEEDS]
        assert set(gates['stress_runs'])==set(stress)
        expected.update(stress)
        assert gates['status']==('STRESS_COMPLETED' if stress else 'STRESS_BLOCKED_PREREQUISITE_NOT_MET')
        for name in confirmed+gates['original_comparison']+stress:
            assert runs[name]['cfg']==selected['cfg'],name
    else:assert selected['status']=='NO_CONFIG_PASSED' and gates['status']=='BLOCKED_NO_CONFIG'
    assert set(runs)==expected,(set(runs)-expected,expected-set(runs))
    for r in runs.values():
        assert r['train']['sha256']!=r['validation']['sha256']
        for entry in r['checkpoints'].values():assert entry['metrics']['passed']==passed(entry['metrics'])
        if r['status']=='complete':
            assert len(r['history'])==r['epochs_completed']
            assert r['updates']==r['epochs_completed']*((r['windows']+63)//64)
            assert set(r['checkpoints'])=={str(e) for e in CHECKS if e<=r['epochs_completed']}
            if r['tiny'] or r['name'].startswith('tune-'):
                earlier=[v for k,v in r['checkpoints'].items() if int(k)<r['epochs_completed']]
                assert not any(passed(v['metrics']) for v in earlier)
    # Same initialization/order and first 100 epochs in tuning vs confirmation s42.
    if candidates:
        assert runs['confirm-compatible-s42']['history']==best['history']
        for seed in SEEDS:
            pairs=[runs[f'original-a0-{g}-s{seed}'] for g in ('SPD','general')]
            assert pairs[0]['window_samples']==pairs[1]['window_samples']
            assert pairs[0]['train']==pairs[1]['train'] and pairs[0]['validation']==pairs[1]['validation']
            assert pairs[0]['shuffle_seed']==pairs[1]['shuffle_seed']
    tests=json.loads((OUT/'test_evaluation.json').read_text())['results']
    expected_tests={n for n,r in runs.items() if not r['tiny'] and not n.startswith('tune-') and r['status']=='complete'}
    assert set(tests)==expected_tests
    for n,entry in tests.items():
        assert entry['data']['sha256'] not in {runs[n]['train']['sha256'],runs[n]['validation']['sha256']}
    recompute=json.loads((OUT/'recompute.json').read_text())
    assert recompute['verdict']=='PASS' and recompute['runs']==len(runs)
    assert recompute['test_evaluations']==len(tests)
    assert recompute['checkpoint_evaluations']==sum(len(r['checkpoints']) for r in runs.values())
    failed=[n for n,r in runs.items() if r['status']!='complete']
    for seed in SEEDS:
        baseline=make_model(seed);general=make_model(seed,True)
        for a,b in zip(matrices(baseline),matrices(general)):
            torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)
    previous=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        general=make_model(42,True)
        batch=(torch.randn(1,2),torch.randn(1,3,1),torch.randn(1,3,2))
        class GeneralLoss(torch.nn.Module):
            def __init__(self):super().__init__();self.module=general
            def forward(self):return loss(self.module,batch)
        wrapped=GeneralLoss()
        key='module.model.operator.core.H_head'
        parameter=dict(wrapped.named_parameters())[key]
        assert torch.autograd.gradcheck(lambda value:torch.func.functional_call(wrapped,{key:value},()),
                                       (parameter,),eps=1e-6,atol=2e-5,rtol=2e-4)
    finally:
        torch.set_default_dtype(previous)
    result={'verdict':'PASS','expected_run_inventory_verified':True,'selection_rederived':True,
            'conditional_gates_rederived':True,'paired_original_windows_verified':True,'general_initial_AB_match_all_seeds':True,'general_head_float64_gradcheck':True,
            'runs':len(runs),'updates':sum(r['updates'] for r in runs.values()),'failed_training_runs':failed,
            'stress_omitted_by_protocol':not gates.get('stress_runs',[]),
            'note':'PASS verifies execution and recomputation; it does not mean the selected configuration met the scientific five-seed gate.'}
    save(OUT/'verification.json',result)
    return runs,selected,gates,tests,result


def report():
    runs,selected,gates,tests,v=verify()
    before=(OUT/'protocol_before_run.md').read_text()
    lines=['', '## Kết quả thực chạy', '',
           f"**Đã thực hiện {v['runs']} run, {v['updates']:,} optimizer updates.** Tất cả run hữu hạn;"]
    if v['failed_training_runs']:lines[-1]=f"**Đã thực hiện {v['runs']} run**, các run lỗi: {v['failed_training_runs']}."
    recompute=json.loads((OUT/'recompute.json').read_text())
    lines += [f"đã nạp lại và đối chiếu {recompute['checkpoint_evaluations']} checkpoint evaluations và {len(tests)} test evaluations.",
        'Inventory, quy tắc chọn config và các điều kiện mở/dừng từng bước được kiểm tra độc lập.', '',
        '**1. Code/đáp án biết trước:** PASS. Forward khớp tuyệt đối; gradient cached so với',
        'module chính thức lệch tối đa 3.73e-9. Sai phân hữu hạn loss theo tham số operator',
        'float64 PASS. Đáp án cài sẵn cho response=0, H15 Euler MSE=2.44e-15;',
        'trên RK4, H15 MSE=2.78e-6. Sai khác solver được ghi riêng. Kiểm tra float64',
        'đặt default dtype trong phạm vi test vì builder upstream tạo zeros không chỉ rõ dtype;',
        'không sửa code upstream và không suy ra toàn bộ hệ thống không còn bug.', '',
        '**2. Overfit nhỏ:** loss gốc, LR=.001 cố định, 1.000 epochs đạt response',
        f"{runs['tiny-lr0.001']['final_validation_or_tiny_train']['response']:.6f}, H15 ratio so với bỏ input",
        f"{runs['tiny-lr0.001']['final_validation_or_tiny_train']['H15_ratio_to_ignore_input']:.6f}. Mô hình học được tác dụng đầu vào trên bài tương thích.", '',
        '**3. Tìm config:** kết quả validation seed42; chọn bằng quy tắc đã khóa, không dùng test.', '',
        '| LR | Scheduler | Epoch dừng | Response | H15/bỏ input | Đạt |', '|---:|---|---:|---:|---:|---|']
    for name in selected['tuning_runs']:
        r=runs[name];m=r['final_validation_or_tiny_train']
        if m:lines.append(f"| {r['cfg']['lr']} | {r['cfg']['scheduler']} | {r['epochs_completed']} | {m['response']:.6g} | {m['H15_ratio_to_ignore_input']:.6g} | {m['passed']} |")
    matched_low=runs['tune-lr0.001-constant-noenc0']['checkpoints']['100']['metrics']['response']
    matched_high=runs['tune-lr0.01-constant-noenc0']['checkpoints']['100']['metrics']['response']
    lines += ['', f'Cùng dữ liệu, khởi tạo, loss và **100 epochs** trên seed42: LR=.001 cho response {matched_low:.6f}, trong khi LR=.01 cho {matched_high:.6f}. Đây là đối chứng trực tiếp cho ảnh hưởng của LR.', '']
    if selected['status']=='SELECTED':
        cfg=selected['cfg']
        lines += ['',f"Config được chọn: **LR={cfg['lr']}, {cfg['scheduler']}, {cfg['epochs']} epochs, loss gốc**. Không cần mở nhánh bỏ encoder loss.",
            'Quy tắc ưu tiên ít updates nhất chọn 100 epochs; việc này không chứng nhận ngân sách đó phù hợp mọi seed.', '',
            '**4. Xác nhận trên 5 seeds:**', '',
            '| Seed | Compatible validation response | Compatible test response | Validation đạt cả hai tiêu chí |', '|---:|---:|---:|---|']
        for seed in SEEDS:
            name=f'confirm-compatible-s{seed}';m=runs[name]['final_validation_or_tiny_train'];t=tests[name]['metrics']
            lines.append(f"| {seed} | {m['response']:.6g} | {t['response']:.6g} | {m['passed']} |")
        n=sum(passed(runs[f'confirm-compatible-s{s}']['final_validation_or_tiny_train']) for s in SEEDS)
        lines += ['',f"**Đạt {n}/5 seeds, chưa đạt điều kiện 5/5.** Seed42 đã dùng để tune; bốn seeds còn lại là xác nhận mới, không gọi cả năm là độc lập với tuning.",
            'Bảng test được tạo sau khi config và quyết định các stage đã khóa; không dùng để chọn lại cấu hình.', '',
            '| G=e1, config/ngân sách đã khóa | Validation response, mean ± sample std | Seeds đạt |', '|---|---:|---:|']
        for geometry in ('SPD','general'):
            group=[runs[f'original-a0-{geometry}-s{s}']['final_validation_or_tiny_train'] for s in SEEDS]
            xs=[m['response'] for m in group]
            lines.append(f"| {geometry} | {statistics.mean(xs):.6g} ± {statistics.stdev(xs):.3g} | {sum(passed(m) for m in group)}/5 |")
    lines += ['', '**5. Stress phi tuyến:** '+('đã thực hiện theo gate.' if gates.get('stress_runs') else '**không chạy**, vì prerequisite 5/5 chưa đạt. Đây là dừng có điều kiện đã khai báo trước, không phải bỏ sót kết quả.'), '',
        '## Điều có thể và chưa thể kết luận', '',
        '- **Config/ngân sách có ảnh hưởng thực:** cùng baseline/loss gốc học được positive control;',
        '  LR=.0001 chưa đạt sau 1000 epochs trên seed tune, trong khi .01 đạt sau100.',
        '- **Chưa chứng nhận config ổn định:** lựa chọn từ một seed chưa vượt gate 5/5.',
        '  Không kết luận rằng mọi config khác đều thất bại; các ứng viên .001/1000 chưa được xác nhận đa seed.',
        '- **Chưa cô lập toàn bộ nguyên nhân trên hệ gốc:** cả SPD và general đều chưa đạt',
        '  ở ngân sách được chọn. Vì vậy chưa thể quy toàn bộ sai số cho SPD hay fixed-B.',
        '- H2/H3 tiếp tục NO-GO; không sửa historical gates và không tuyên bố tái lập paper.',
        '- Bước tiếp theo có cơ sở: protocol mới xác nhận ngân sách dài hơn và chọn config',
        '  bằng nhiều validation seeds; khóa test mới nếu tiếp tục chọn cấu hình. Không tự mở rộng',
        '  tìm kiếm sau khi thấy test của đợt này.', '',
        '## Chạy lại và bằng chứng', '', '```bash',
        'MPLCONFIGDIR=/tmp/cartpole-mpl OMP_NUM_THREADS=1 .venv/bin/python -m research.diagnostics.p2_config_check --stage all',
        'MPLCONFIGDIR=/tmp/cartpole-mpl OMP_NUM_THREADS=1 .venv/bin/python -m research.diagnostics.p2_config_check --stage recompute',
        '.venv/bin/python -m research.diagnostics.report_p2_config_check', '```', '',
        'Lệnh all tái sử dụng run đã hoàn tất; recompute nạp lại checkpoint. Muốn train lại từ đầu,',
        'dùng bản sao script/output riêng để giữ bằng chứng gốc. Protocol trước khi chạy, source/data/',
        'checkpoint hashes, history LR/loss và gradient từng thành phần được lưu tại',
        '[thư mục bằng chứng](../../outputs/diagnosis/p2_config_check_2026-11-16/).', '',
        '[Kiểm chứng độc lập](../../outputs/diagnosis/p2_config_check_2026-11-16/verification.json) ·',
        '[Config đã chọn](../../outputs/diagnosis/p2_config_check_2026-11-16/selected_config.json) ·',
        '[Gates](../../outputs/diagnosis/p2_config_check_2026-11-16/stage_gates.json).', '']
    if (OUT/'tests.json').exists():
        count=json.loads((OUT/'tests.json').read_text())['tests']
        lines += [f'Bộ kiểm thử repository: **{count} tests PASS**.', '']
    DOC.write_text(before+'\n'.join(lines))
    print(json.dumps(v,indent=2))


if __name__=='__main__':report()
