process init_windows {
  input:
  path patent_dir
  path settings_file
  path output_dir

  output:
  path("${output_dir}/windows/*.npy")


  """
  echo "$output_dir/windows"
  init_windows.py --settings $settings_file --patent_dir $patent_dir --output_dir $output_dir
  """
}

process create_embeddings {

  input:
  path window_file
  val output_file
  path model_file
  path patent_dir

  output:
  path output_file

  // model_name = model_file.getSimpleName()
  // println model_name
  // window_name = window_file.getSimpleName()


  """
  create_embeddings.py --patent_dir $patent_dir --window_fp $window_file --model_fp $model_file --output_fp $output_file
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
  window_ch = Channel.fromPath(params.output_dir + "/windows/*.npy").map {[it, it.simpleName]}
  combined_ch = window_ch.combine(model_ch).map {[it[0],
    it[2], params.output_dir + "/modelres/" + it[2].simpleName + "/" + it[1] + "/embedding.npy"]}
  combined_ch.multiMap {it -> 
                        window_file: it[0]
                        model_file: it[1]
                        output_file: it[2]}.set{res}
  create_embeddings(res.window_file, res.output_file, res.model_file, patent_ch)
}