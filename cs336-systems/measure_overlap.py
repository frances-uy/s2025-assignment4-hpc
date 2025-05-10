from hta.trace_analysis import TraceAnalysis

# Analyze naive DDP trace
naive_trace_path = "cs336-systems/gn-16-30-00_598906.1746786828134671677.pt.trace.json"
naive_analysis = TraceAnalysis(naive_trace_path)
naive_overlap = naive_analysis.comm_overlap_ratio()
print(f"Naive DDP overlap ratio: {naive_overlap:.2f}")

# Analyze overlap DDP trace
overlap_trace_path = "cs336-systems/gn-16-30-00_599384.1746786841539020869.pt.trace.json"
overlap_analysis = TraceAnalysis(overlap_trace_path)
overlap_score = overlap_analysis.comm_overlap_ratio()
print(f"Overlap DDP overlap ratio: {overlap_score:.2f}")
