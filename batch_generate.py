#!/usr/bin/env python3
"""
批量生成符合 BIRD 评测格式的 SQL 预测文件（使用完整 XiYan-SQL 架构）
"""

import os
import json
import sys
from tqdm import tqdm
from bird_dataset import BirdDataset
from generate_sql import XiYanSQLGenerator
from config import XiYanSQLConfig


def generate_predict_file(
    dataset_path: str,
    output_path: str,
    api_key: str = None,
    use_schema_filter: bool = True,
    use_multi_generator: bool = True,
    model: str = "qwen3.6-plus",
    num_samples: int = None,
    start_index: int = 0,
    save_prompts: bool = True,
    prompt_output_dir: str = "./prompts",
    max_iterations: int = 2
):
    """
    批量生成 SQL 并保存为 BIRD 评测格式
    
    Args:
        dataset_path: BIRD 数据集路径
        output_path: 输出 JSON 文件路径
        api_key: 阿里云 API Key
        use_schema_filter: 是否使用 SchemaFilter
        use_multi_generator: 是否使用多生成器集成
        model: 使用的模型
        num_samples: 生成的样本数（None 表示全部）
        start_index: 开始的样本索引
        save_prompts: 是否保存 prompt 到文件
        prompt_output_dir: prompt 输出目录
        max_iterations: Schema Filter 迭代次数
    """
    # 加载数据集
    print(f"📥 加载数据集: {dataset_path}")
    dataset = BirdDataset(dataset_path)
    
    # 确定要处理的样本范围
    samples = dataset.samples
    if num_samples is not None:
        samples = samples[start_index:start_index + num_samples]
    else:
        samples = samples[start_index:]
    
    print(f"📊 总样本数: {len(dataset.samples)}")
    print(f"🎯 处理样本: {start_index} ~ {start_index + len(samples) - 1}")
    print(f"🔍 使用 SchemaFilter: {use_schema_filter}")
    print(f"🤖 使用 Multi-Generator: {use_multi_generator}")
    print(f"🤖 模型: {model}")
    print(f"💾 保存 Prompt: {save_prompts}")
    
    # 数据库路径
    db_path = os.path.join(dataset_path, "dev_databases", f"{samples[0].db_id}", f"{samples[0].db_id}.sqlite")
    
    # 初始化生成器
    generator = XiYanSQLGenerator(api_key=api_key, model=model, db_path=db_path)
    
    # 创建 prompt 输出目录
    if save_prompts and not os.path.exists(prompt_output_dir):
        os.makedirs(prompt_output_dir)
    
    # 生成结果
    predict_dict = {}
    
    for i, sample in enumerate(tqdm(samples, desc="生成 SQL")):
        idx = start_index + i
        
        try:
            # 获取 schema
            full_schema = dataset.get_schema(sample.db_id)
            
            # 生成 SQL
            result = generator.generate_sql(
                question=sample.question,
                full_schema=full_schema,
                evidence=sample.evidence,
                use_schema_filter=use_schema_filter,
                use_multi_generator=use_multi_generator,
                max_iterations=max_iterations
            )
            
            # 保存 prompt 到文件
            if save_prompts:
                filter_suffix = "_filtered" if use_schema_filter else "_full"
                multi_suffix = "_multi" if use_multi_generator else "_single"
                prompt_file = os.path.join(prompt_output_dir, f"sample_{sample.question_id}{filter_suffix}{multi_suffix}.txt")
                
                with open(prompt_file, 'w', encoding='utf-8') as f:
                    f.write("=" * 80 + "\n")
                    f.write(f"样本 #{sample.question_id}\n")
                    f.write(f"数据库: {sample.db_id}\n")
                    f.write(f"SchemaFilter: {use_schema_filter}\n")
                    f.write(f"MultiGenerator: {use_multi_generator}\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(f"问题:\n{sample.question}\n\n")
                    if sample.evidence:
                        f.write(f"证据:\n{sample.evidence}\n\n")
                    f.write(f"真实 SQL:\n{sample.sql}\n\n")
                    
                    if 'candidates' in result:
                        f.write("=" * 80 + "\n")
                        f.write(f"Generated Candidates ({len(result['candidates'])}):\n")
                        f.write("=" * 80 + "\n\n")
                        for j, cand in enumerate(result['candidates']):
                            f.write(f"Candidate {j+1} [{cand['source']}]:\n")
                            f.write(f"```sql\n{cand['sql']}\n```\n\n")
                    
                    if 'reorganized_candidates' in result:
                        f.write("=" * 80 + "\n")
                        f.write(f"Reorganized Candidates:\n")
                        f.write("=" * 80 + "\n\n")
                        for j, cand in enumerate(result['reorganized_candidates']):
                            f.write(f"Candidate {j+1} [{cand['source']}]\n")
                            f.write(f"```sql\n{cand['sql']}\n```\n\n")
                    
                    f.write("=" * 80 + "\n")
                    f.write(f"Final Selected SQL:\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(f"```sql\n{result.get('sql', '(None)')}\n```")
            
            # 格式化：{sql}\t----- bird -----\t{db_name}
            sql_str = result.get("sql", "").strip() if result.get("sql") else ""
            predict_value = f"{sql_str}\t----- bird -----\t{sample.db_id}"
            
            predict_dict[str(idx)] = predict_value
            
        except Exception as e:
            print(f"\n⚠️  样本 {idx} 出错: {e}")
            import traceback
            traceback.print_exc()
            predict_dict[str(idx)] = f" \t----- bird -----\t{sample.db_id}"
    
    # 保存结果
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predict_dict, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 结果已保存到: {output_path}")
    print(f"📝 生成了 {len(predict_dict)} 个预测")
    if save_prompts:
        print(f"💾 Prompts 已保存到: {prompt_output_dir}")


def main():
    """主函数"""
    # 配置
    DATASET_PATH = r"D:\download\dev_20240627"
    API_KEY = "sk-bb901ef8d7e44cb0be1c535e137974c4"
    
    print("=" * 80)
    print("BIRD 批量 SQL 生成工具（XiYan-SQL 完整架构）")
    print("=" * 80)
    
    # 交互式选择
    print("\n请选择模式:")
    print("  1. XiYan-SQL 完整模式（SchemaFilter + Multi-Generator）")
    print("  2. 仅 SchemaFilter + 单生成器")
    print("  3. 完整 Schema + 单生成器")
    print("  4. 三者都生成（对比）")
    
    mode = input("\n请输入选项 (1-4): ").strip()
    
    # 选择样本数量
    print("\n请选择生成样本数:")
    print("  0. 前 1 个（快速测试）")
    print("  1. 前 10 个（快速测试）")
    print("  2. 前 50 个")
    print("  3. 全部样本")
    print("  4. 自定义范围")
    
    sample_mode = input("\n请输入选项 (0-4): ").strip()
    
    num_samples = None
    start_index = 0
    if sample_mode == "0":
        num_samples = 1
    elif sample_mode == "1":
        num_samples = 10
    elif sample_mode == "2":
        num_samples = 50
    elif sample_mode == "4":
        start_index = int(input("\n请输入开始索引: ").strip())
        num_samples = int(input("请输入样本数: ").strip())
    
    # 输出目录
    output_dir = input("\n请输入输出目录 (默认 ./predictions): ").strip()
    if not output_dir:
        output_dir = "./predictions"
    
    # 是否保存 prompt
    save_prompts_input = input("\n是否保存 Prompt 到文件? (y/n, 默认 y): ").strip().lower()
    save_prompts = save_prompts_input != "n"
    
    # 执行生成
    if mode == "1":
        output_path = os.path.join(output_dir, "predict_dev_xiyan.json")
        prompt_output_dir = os.path.join(output_dir, "prompts_xiyan")
        print(f"\n🚀 生成（XiYan-SQL 完整模式）...")
        generate_predict_file(
            dataset_path=DATASET_PATH,
            output_path=output_path,
            api_key=API_KEY,
            use_schema_filter=True,
            use_multi_generator=True,
            num_samples=num_samples,
            start_index=start_index,
            save_prompts=save_prompts,
            prompt_output_dir=prompt_output_dir
        )
    elif mode == "2":
        output_path = os.path.join(output_dir, "predict_dev_filtered.json")
        prompt_output_dir = os.path.join(output_dir, "prompts_filtered")
        print(f"\n🚀 生成（仅 SchemaFilter）...")
        generate_predict_file(
            dataset_path=DATASET_PATH,
            output_path=output_path,
            api_key=API_KEY,
            use_schema_filter=True,
            use_multi_generator=False,
            num_samples=num_samples,
            start_index=start_index,
            save_prompts=save_prompts,
            prompt_output_dir=prompt_output_dir
        )
    elif mode == "3":
        output_path = os.path.join(output_dir, "predict_dev_full.json")
        prompt_output_dir = os.path.join(output_dir, "prompts_full")
        print(f"\n🚀 生成（完整 Schema）...")
        generate_predict_file(
            dataset_path=DATASET_PATH,
            output_path=output_path,
            api_key=API_KEY,
            use_schema_filter=False,
            use_multi_generator=False,
            num_samples=num_samples,
            start_index=start_index,
            save_prompts=save_prompts,
            prompt_output_dir=prompt_output_dir
        )
    elif mode == "4":
        output_path1 = os.path.join(output_dir, "predict_dev_xiyan.json")
        output_path2 = os.path.join(output_dir, "predict_dev_filtered.json")
        output_path3 = os.path.join(output_dir, "predict_dev_full.json")
        prompt_output_dir1 = os.path.join(output_dir, "prompts_xiyan")
        prompt_output_dir2 = os.path.join(output_dir, "prompts_filtered")
        prompt_output_dir3 = os.path.join(output_dir, "prompts_full")
        
        print(f"\n🚀 生成（XiYan-SQL 完整模式）...")
        generate_predict_file(
            dataset_path=DATASET_PATH,
            output_path=output_path1,
            api_key=API_KEY,
            use_schema_filter=True,
            use_multi_generator=True,
            num_samples=num_samples,
            start_index=start_index,
            save_prompts=save_prompts,
            prompt_output_dir=prompt_output_dir1
        )
        
        print(f"\n🚀 生成（仅 SchemaFilter）...")
        generate_predict_file(
            dataset_path=DATASET_PATH,
            output_path=output_path2,
            api_key=API_KEY,
            use_schema_filter=True,
            use_multi_generator=False,
            num_samples=num_samples,
            start_index=start_index,
            save_prompts=save_prompts,
            prompt_output_dir=prompt_output_dir2
        )
        
        print(f"\n🚀 生成（完整 Schema）...")
        generate_predict_file(
            dataset_path=DATASET_PATH,
            output_path=output_path3,
            api_key=API_KEY,
            use_schema_filter=False,
            use_multi_generator=False,
            num_samples=num_samples,
            start_index=start_index,
            save_prompts=save_prompts,
            prompt_output_dir=prompt_output_dir3
        )
        
        print(f"\n📁 对比文件:")
        print(f"   XiYan-SQL 完整: {output_path1}")
        print(f"   仅 SchemaFilter: {output_path2}")
        print(f"   完整 Schema: {output_path3}")
        if save_prompts:
            print(f"   Prompts(XiYan): {prompt_output_dir1}")
            print(f"   Prompts(过滤): {prompt_output_dir2}")
            print(f"   Prompts(完整): {prompt_output_dir3}")
    
    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
