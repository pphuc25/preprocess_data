from datasets import load_dataset
import argparse
import re


dataset = load_dataset('pphuc25/wikipedia-vietnam', split='train')

def config():
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('--dataset_name', type=str, default='pphuc25/wikipedia-vietnam',
                                help='')
    parser.add_argument('--split', type=str, default='train',
                                help='')
    parser.add_argument('--token', type=str, default='None',
                                help='')
    
    args = parser.parse_args()
    return args

class PreprocessData:
    def __init__(self) -> None:
        self.pt_title2, self.pt_title3 = r"''(.*?)''", r"'''(.*?)'''"
        self.pt_h2, self.pt_h3 = r'==.*?==\s*', r'===.*?===\s*'
        self.pt_get_categories = r'Thể loại:([^Thể].*?)(?=\s*Thể loại:|$)'
        self.pt_remove_categories = r'Thể loại:.*?(?=\s*Thể loại:|$)'

    def create_title_column(self, text):
        """
        Get the first word which like '''Ngo Tat To''' to title, in case there's
        no word like this, the title return None
        """
        match = re.search(self.pt_title3, text) 
        if match: return match.group(1)
        return None

    @staticmethod
    def remove_pt_title(pattern_type, text):
        """
        Remove any synctax that have pattern '' (The length of '' can be
        flexible, such as they can be ''', '', or ')
        Example: '''Ngo Tat To''' will become Nguyen Tat To
        """
        new_text = re.sub(pattern_type, r"\1", text)
        return new_text

    @staticmethod
    def remove_pt_h(pattern_type, text):
        """
        Remove any word that have pattern `=== content ===` (the length of =
        can be flexible, such as they can be `==`, = or `====`).
        Example: `=== Vai tro ===` will be remove, `== Vai tro ==` will be remove
        """
        new_text = re.sub(pattern_type, '', text, flags=re.MULTILINE)
        return new_text

    def create_categories_column(self, text):
        """
        Find the pattern `Thể loại:` and get the content after it
        """
        categories = re.findall(self.pt_get_categories, text)
        return categories
    
    def remove_categories_in_text(self, text):
        """
        Remove all the content that related to `Thể loại:`
        """
        new_text = re.sub(self.pt_remove_categories, '', text)
        return new_text


def main():
    args = config()
    preprocess_method = PreprocessData()
    dataset = load_dataset(args.dataset_name, split=args.split, use_auth_token=args.token)

    # Flow pipeline preprocess
    def preprocess_mapping(dataset):
        text = dataset['text']
        dataset['title'] = preprocess_method.create_title_column(text)
        new_text = preprocess_method.remove_pt_title(preprocess_method.pt_title3, text)
        new_text = preprocess_method.remove_pt_title(preprocess_method.pt_title2, new_text)

        new_text = preprocess_method.remove_pt_h(preprocess_method.pt_h3, new_text)
        new_text = preprocess_method.remove_pt_h(preprocess_method.pt_h2, new_text)
        
        dataset['categories'] = ", ".join(preprocess_method.create_categories_column(new_text))
        new_text = preprocess_method.remove_categories_in_text(new_text)
        
        dataset['text'] = new_text.strip()
        return dataset

    # dataset = dataset.select(range(100))
    new_dataset = dataset.map(preprocess_mapping)
    print(new_dataset[0])
    return new_dataset

if __name__ == "__main__":
    main()