# Results preservation and server checkout retirement

The owner requested verification that results are on GitHub, followed by deletion of only `/workspace-SR003.nfs2/ARG-GNN` on anogena. The campaign processes have been stopped. The deletion script requires an identical local and GitHub `main` commit, unchanged tracked files, verified archive hashes, the exact original directory identity and no nested mounts or active processes using the directory. It will remove only that exact server folder.

## Preserved results

The published six-class comparison, direct four-class comparison, independent review, manuscript, code and existing Git history remain versioned at their original paths. Existing uncommitted tracked records are saved in this retirement commit without changing scores or claiming benchmark completion.

Additional evidence is stored in 67 `evidence_*.tar.gz` files: **38,260 files**, **1,353,540,334 compressed bytes**. This includes **1,227 small model and prediction artifacts**, historical uncommitted source/configuration files, experiment records and operational metadata. The four current TreeLoom model variants include the European six-class and four-class fits, the American fit, and the ARGMix-setting fit. Available native comparator predictions and the small selected ARGMix model are also preserved. These files are historical evidence, not additional certified results.

Every archive member was read back and verified against its recorded SHA256 and size. The full inventories are `archive_manifest.jsonl.gz` and `additional_manifest.jsonl.gz`; each entry maps an original project-relative path to an archive and checksum. `archive_summary.json` records checksums for the archive files. `artifact_selection.json` and `additional_artifacts.json` list the selected binary outputs.

For recovery, clone this repository, inspect the archive manifests, and extract the needed archives into the clone root. Archives preserve original relative paths, including selected historical `.runtime/benchmarking` records. Review historical operational scripts before using them; archiving does not approve their execution. Tracked files and current scientific score tables take precedence over old experimental snapshots.

## Explicit exclusions

This is a results and source archive, **not a full backup of the allocation directory**. Large raw simulations, inputs, prepared caches, software environments, private credentials, files over 20 MiB that were not already tracked, large competitor checkpoints and transient raw logs are excluded. In particular, the interrupted multi-gigabyte ARGformer TEST009 optimizer checkpoint is not archived. Its training cannot resume from that server checkpoint after deletion. Data must be regenerated and unavailable model states retrained if the study is resumed.

Deletion of the folder is authorized, but this committed record is prepared before deletion. The actual deletion outcome is reported in the chat only after filesystem verification. The removal must not touch the Mac workspace, other projects or unrelated processes.
