from dataclasses import dataclass, field
@dataclass
class Metrics:
    ingested:        int = 0
    skipped_size:    int = 0
    skipped_symlink: int = 0
    skipped_dotfile: int = 0
    skipped_confined:int = 0
    skipped_inode:   int = 0
    encoding_warns:  int = 0
    errors:          int = 0
    by_type: dict    = field(default_factory=lambda: {
        ".txt": 0, ".csv": 0, ".pdf": 0,
        "dailydialog": 0, "personachat": 0,
    })

    def report(self) -> dict:
        return {
            "ingested":         self.ingested,
            "skipped_size":     self.skipped_size,
            "skipped_symlink":  self.skipped_symlink,
            "skipped_dotfile":  self.skipped_dotfile,
            "skipped_confined": self.skipped_confined,
            "skipped_inode":    self.skipped_inode,
            "encoding_warns":   self.encoding_warns,
            "errors":           self.errors,
            "by_type":          self.by_type,
        }
