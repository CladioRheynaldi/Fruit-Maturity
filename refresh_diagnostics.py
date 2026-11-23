import json
from train import OUT, diagnostics

def main():
    summary_path = OUT / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    report_path = OUT / 'REPORT.md'
    report = report_path.read_text(encoding='utf-8')
    for run in summary['experiments']:
        old = run['diagnostics']
        run['diagnostics'] = diagnostics(run['history'])
        for before, after in zip(old, run['diagnostics']):
            def line(d):
                return f"- {d['topic']}: {'indicator present' if d['flag'] else 'indicator absent'}. {d['detail']}"
            report = report.replace(line(before), line(after), 1)
        result_path = OUT / run['config']['id'] / 'results.json'
        result = json.loads(result_path.read_text(encoding='utf-8'))
        result['diagnostics'] = run['diagnostics']
        result_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    report_path.write_text(report, encoding='utf-8')
    print('Diagnostics refreshed from saved epoch history; model weights and metrics unchanged.')

if __name__ == '__main__':
    main()
