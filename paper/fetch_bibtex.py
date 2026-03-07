import requests
import time
import re

# 你的文献列表 (直接粘贴在这里)
raw_refs = """
1. Reference sequence (RefSeq) database at NCBI: current status, taxonomic expansion, and functional annotation (O'Leary et al.)
2. Transformer Models for Predicting Bacteriophage-Host Relationships (Azam et al.)
3. A Reference Viral Database (RVDB) To Enhance Bioinformatics Analysis of High-Throughput Sequencing for Novel Virus Detection (Goodacre et al.)
4. A comprehensive and quantitative exploration of thousands of viral genomes (Mahmoudabadi et al.)
5. A machine learning approach to predict strain-specific phage-host interactions (Camejo et al.)
6. A network-based integrated framework for predicting virus–prokaryote interactions (Wang et al.)
7. Advanced Strategies in Phage Research: Innovations, Applications, and Challenges (Wu et al.)
8. Advances in phage–host interaction prediction: in silico method enhances the development of phage therapies (Nie et al.)
9. An ensemble method for prediction of phage-based therapy against bacterial infections (Aggarwal et al.)
10. Bacteriophages presence in nature and their role in the natural selection of bacterial populations (Naureen et al.)
11. Biological challenges of phage therapy and proposed solutions: a literature review (Lawrence & Baldridge)
12. Classifying the Lifestyle of Metagenomically-Derived Phages Sequences Using Alignment-Free Methods (Song)
13. Codon Bias is a Major Factor Explaining Phage Evolution in Translationally Biased Hosts (Lucks et al.)
14. Comprehensive Bibliographic Analysis of Bacteriophage-Host Interaction Prediction: From Biological Foundations to Neural Architectures (Review)
15. Computational Prediction of Bacteriophage Host Ranges (Versoza & Pfeifer)
16. Computational approaches for virus host prediction: A review of methods and applications (Shang et al.)
17. Computational approaches to predict bacteriophage–host relationships (Edwards et al.)
18. Deep Learning Transforms Phage-Host Interaction Discovery from Metagenomic Data (Yang et al., Preprint)
19. DeepHost: Phage host prediction with convolutional neural network (Wang et al.)
20. Ecogenomics and potential biogeochemical impacts of globally abundant ocean viruses (Roux et al.)
21. Effects of bacteriophages on gut microbiome functionality (Kurilovich & Geva-Zatorsky)
22. Evolutionarily Conserved Orthologous Families in Phages Are Relatively Rare in Their Prokaryotic Hosts (Kristensen et al.)
23. From genomic signals to prediction tools: a critical feature analysis and rigorous benchmark for phage–host prediction (Shang et al.)
24. Genome Landscapes and Bacteriophage Codon Usage (Lucks et al.)
25. Global features of sequences of bacterial chromosomes, plasmids and phages revealed by analysis of oligonucleotide usage patterns (Reva & Tümmler)
26. HostPhinder: A Phage Host Prediction Tool (Villarroel et al.)
27. How phage therapy supports One Health in the AMR fight (Feature Article)
28. IMG/VR v3: an integrated ecological and evolutionary framework for interrogating genomes of uncultivated viruses (Roux et al.)
29. IMG/VR v4: an expanded database of uncultivated virus genomes within a framework of extensive functional, taxonomic, and ecological metadata (Camargo et al.)
30. Improving gut virome comparisons using predicted phage host information (Shamash et al.)
31. Investigations of Oligonucleotide Usage Variance Within and Between Prokaryotes (Bohlin et al.)
32. Marine Bacteriophages as Next-Generation Therapeutics: Insights into Antimicrobial Potential and Application (Banicod et al.)
33. Maritime Viruses (Ohio Supercomputer Center Feature)
34. MoEPH: an adaptive fusion-based LLM for predicting phage-host interactions in health informatics (Chen et al.)
35. Oligonucleotide correlations between infector and host genomes hint at evolutionary relationships (Barrai & Scapoli)
36. PB-LKS: a python package for predicting phage–bacteria interaction through local K-mer strategy (Qiu et al.)
37. PHISDetector: A Tool to Detect Diverse In Silico Phage–host Interaction Signals for Virome Studies (Zhou et al.)
38. PHPGAT: predicting phage hosts based on multimodal heterogeneous knowledge graph with graph attention network (Liu et al.)
39. Phage-Host Prediction Using a Computational Tool Coupled with 16S rRNA Gene Amplicon Sequencing (Andrianjakarivony et al.)
40. Phages and Human Health: More Than Idle Hitchhikers (Lawrence & Baldridge)
41. Phages in the Gut Ecosystem (Zuppi et al.)
42. Prokaryotic Virus Orthologous Groups (pVOGs): a resource for comparative genomics and protein family annotation (Grazziotin et al.)
43. Prokaryotic virus host predictor: a Gaussian model for host prediction of prokaryotic viruses in metagenomics (Lu et al., PHP tool)
44. RNAVirHost: a machine learning–based method for predicting hosts of RNA viruses through viral genomes (Chen et al.)
45. Role of bacteriophages in shaping gut microbial community (Mahmud et al.)
46. Sequence-Based Bioinformatics Approaches to Predict Virus-Host Relationships in Archaea and Eukaryotes (Li)
47. Streamlining CRISPR spacer-based bacterial host predictions to decipher the viral dark matter (Dion et al.)
48. The Role of Bacteriophages in the Gut Microbiota: Implications for Human Health (Emencheta et al.)
49. The use of genomic signature distance between bacteriophages and their hosts displays evolutionary relationships and phage growth cycle determination (Deschavanne & DuBow)
50. Thinking Phage Innovations Through Evolution and Ecology (Brives et al.)
51. Unveiling the potential bacteriophage therapy: a systematic review (Ibrahim et al.)
52. WIsH: who is the host? Predicting prokaryotic hosts from metagenomic phage contigs (Galiez et al.)
53. vHULK, a New Tool for Bacteriophage Host Prediction Based on Annotated Genomic Features and Neural Networks (Amgarten et al.)
"""
# (注：为了演示简洁，我只放了部分。实际使用时请把你的53条全部粘贴到上面的字符串中)

def get_bibtex(query):
    # 清洗查询词：去掉序号和末尾的括号作者，提高搜索准确率
    # 例如 "1. Title (Author)" -> "Title"
    clean_query = re.sub(r'^\d+\.\s*', '', query) # 去掉开头序号
    clean_query = re.sub(r'\s*\(.*?\)$', '', clean_query) # 去掉末尾括号作者
    
    url = "https://api.crossref.org/works"
    params = {
        "query.bibliographic": clean_query,
        "rows": 1, # 只取匹配度最高的第一条
        "select": "DOI,title,author,issued,URL"
    }
    
    try:
        # 1. 搜索 DOI
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        
        if data['message']['items']:
            item = data['message']['items'][0]
            doi = item.get('DOI')
            title = item.get('title', [''])[0]
            print(f"Found: {doi} -> {title[:50]}...")
            
            # 2. 利用 DOI 获取标准 BibTeX
            # content negotiation: 请求 text/bibliography 格式
            headers = {"Accept": "application/x-bibtex"}
            bib_r = requests.get(f"https://doi.org/{doi}", headers=headers, timeout=10)
            
            if bib_r.status_code == 200:
                return bib_r.text
    except Exception as e:
        print(f"Error fetching {query[:30]}: {e}")
    
    return None

# 主程序
all_bibtex = []
refs = [line.strip() for line in raw_refs.strip().split('\n') if line.strip()]

print(f"开始处理 {len(refs)} 条文献...")

for ref in refs:
    bib = get_bibtex(ref)
    if bib:
        all_bibtex.append(bib)
    else:
        # 如果找不到，生成一个占位符，方便手动修改
        print(f"Warning: Could not find match for {ref[:30]}")
        all_bibtex.append(f"% FAILED TO FIND: {ref}\n")
    
    # 礼貌性延时，避免被 API 封禁
    time.sleep(0.5)

# 保存结果
with open("references.bib", "w", encoding="utf-8") as f:
    f.write("\n".join(all_bibtex))

print("完成！已保存为 references.bib")