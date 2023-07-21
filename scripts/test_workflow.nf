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
  publishDir "$params.output_ch/modelres/$model_name/$window_name"

  input:
  tuple path(model_file), path(window_file), val(model_name), val(window_name)
  path patent_dir

  output:
  path "embedding.npy"

  // model_name = model_file.getSimpleName()
  // println model_name
  // window_name = window_file.getSimpleName()


  """
  create_embeddings.py --patent_dir $patent_dir --window_fp $window_file --model_fp $model_file --output_fp embedding.npy
  """
}

process create_novelty_impact {
  input:
  path embedding_file
  path window_file
  path output_file

  output:
  val window_file
  path output_file

  """
  create_novelty_impact.py --output_csv $output_file --embedding_fp $embedding_file --window_fp $window_file
  """
}


params.output_dir = "../data/output"
params.output_ch = Channel.fromPath("../data/output")
params.settings_file = "test-settings.json"
params.patent_dir = "../data/raw/unprocessed"
params.model_file = "tfidf.json"

workflow {
  patent_ch = Channel.fromPath(params.patent_dir)
  settings_ch = Channel.fromPath(params.settings_file)
  // output_ch = Channel.fromPath(params.output_dir)
  model_ch = Channel.fromPath(params.model_file)
  window_path_ch = init_windows(patent_ch, settings_ch, params.output_ch)
  window_ch = Channel.fromPath(params.output_dir + "/windows/*.npy").map {[it, it.simpleName]}
  combined_ch = model_ch.combine(window_ch).map {[it[0], it[1], it[0].simpleName, it[1].simpleName]}
  combined_ch.view()
    // window_file: 
  // )
  // combined_ch = window_ch.combine(model_ch).map {[it[0],
    // it[2], params.output_dir + "/modelres/" + it[2].simpleName + "/" + it[1] + "/embedding.npy"]}
  // combined_ch.multiMap {it -> 
                        // window_file: it[0]
                        // model_file: it[1]
                        // output_file: it[2]}.set{res}
  embedding_ch = create_embeddings(combined_ch, patent_ch)
  // embedding_ch = create_embeddings(res.window_file, res.output_file, res.model_file, patent_ch)
  embedding_ch.view()
  // embedding_ch.multiMap {

  // }
  // create_novelty_impact()
}