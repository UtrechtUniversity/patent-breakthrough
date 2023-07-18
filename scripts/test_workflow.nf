process init_windows {
  input:
  path patent_dir
  path settings_file
  path output_dir

  output:
  path("${output_dir}/windows/*.npy")


  """
  echo "$output_dir/windows"
  init_windows.py --settings $settings_file
  """
}

process create_embeddings {

  input:
  path window_file
  path output_dir
  path model_file

  output:
  path "$output_dir/modelres/$model_name/$window_name/embedding.npy"

  // model_name = model_file.getSimpleName()
  // println model_name
  // window_name = window_file.getSimpleName()


  """
  echo "?"
  """
}


params.output_dir = "../data/output"
params.settings_file = "test-settings.json"
params.patent_dir = "../data/raw/unprocessed"
params.model_file = "tfidf.json"

workflow {
  patent_ch = Channel.fromPath(params.patent_dir)
  settings_ch = Channel.fromPath(params.settings_file)
  output_ch = Channel.fromPath(params.output_dir)
  model_ch = Channel.fromPath(params.model_file)
  window_path_ch = init_windows(patent_ch, settings_ch, output_ch)
  window_ch = Channel.fromList(window_path_ch).map { it -> it.getSimpleName() }.view()
//  window_ch = init_windows.out.map { it -> it.getSimpleName() }.view()
  create_embeddings(init_windows.out, output_ch, model_ch).view()
}