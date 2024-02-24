"""
This is a simplified implementation of the paper Retrieval-Augmented Multimodal
Language Modeling by Yasunaga et al.
"""

# pylint: disable=invalid-name,unused-import, abstract-method

import warnings

import torch
from datasets import load_dataset, load_from_disk
from PIL import Image
from torch import nn
from torch.nn import functional as F
from transformers import CLIPModel, CLIPProcessor

warnings.filterwarnings("ignore")


class Retriever(nn.Module):
    """
    This is essentially a model that can encode both image and text. CLIP was
    used in this paper, so it's default here.
    """

    def __init__(
        self,
        model: str = "openai/clip-vit-base-patch16",
        external_document: str = "mozci/tinysketch",
        from_huggingface: bool = True,
    ):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(model)
        self.model = CLIPModel.from_pretrained(model)
        self.indexed_external_document = self.load_external_documents(
            external_document, from_huggingface
        )

    def get_latent_embeddings(
        self,
        image: Image.Image = None,
        text: str = None,
    ) -> torch.Tensor:
        """
        This method encodes the image and text to their latent embeddings.

        Args:
            image (Image.Image, optional): Image to be encoded.Defaults to None.
            text (str, optional): Text to be encoded. Defaults to None.

        Raises:
            ValueError: Raised when neither image nor text is provided.

        Returns:
            torch.Tensor: Latent embeddings of image or text or image and text.
        """
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding=True,
        )
        if image is not None and text is not None:
            image_latent_embeddings = self.model.get_image_features(
                pixel_values=inputs["pixel_values"]
            )
            text_latent_embeddings = self.model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            norm_image_latent_embeddings = F.normalize(
                image_latent_embeddings, p=2, dim=1
            )
            norm_text_latent_embeddings = F.normalize(
                text_latent_embeddings, p=2, dim=1
            )
            latent_embeddings = (
                norm_image_latent_embeddings + norm_text_latent_embeddings / 2
            )
            return latent_embeddings
        if image is not None:
            image_latent_embeddings = self.model.get_image_features(
                pixel_values=inputs["pixel_values"]
            )
            norm_image_latent_embeddings = F.normalize(
                image_latent_embeddings, p=2, dim=1
            )
            return norm_image_latent_embeddings
        if text is not None:
            text_latent_embeddings = self.model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            norm_text_latent_embeddings = F.normalize(
                text_latent_embeddings, p=2, dim=1
            )
            return norm_image_latent_embeddings
        raise ValueError("Either image or text of both must be provided")

    def forward(
        self,
        image: Image.Image = None,
        text: str = None,
    ) -> torch.Tensor:
        """
        This method retrieves the most similar documents to the query which can
        consist of an image, text or both.

        Args:
            image (Image.Image, optional): Query image. Defaults to None.
            text (str, optional): Query text. Defaults to None.

        Returns:
            torch.Tensor: Retrieved documents
        """
        query_latent_embeddings = self.get_latent_embeddings(
            image, text
        ).detach().numpy()
        _,similar_documents = self.indexed_external_document.get_nearest_examples(
            "latent_embeddings",
            query_latent_embeddings,
            k=5,
        )
        return similar_documents

    def save_external_documents(self, folder_path: str) -> None:
        """
        This method saves the external documents to disk.

        Args:
            folder_path (str): Path to the folder where the external documents
            will be saved.
        """
        self.indexed_external_document.save_faiss_index(
            "latent_embeddings",
            f"{folder_path}/multimodal.faiss",
        )
        self.indexed_external_document.drop_index("latent_embeddings")
        self.indexed_external_document.save_to_disk(
            f"{folder_path}/tiny_sketches_with_latent_embeddings"
        )

    def load_external_documents(
        self, external_document: str, from_huggingface: bool
    ) -> None:
        """
        This method loads the external documents from disk.

        Args:
            external_document (str): Path to the folder where the external
            documents are saved.
        Returns:
            : Retrieved documents
        """
        if from_huggingface:
            external_document = load_dataset(
                external_document,
                split="train",
            )
            indexed_external_document = external_document.map(
                lambda x: {
                    "latent_embeddings": self.get_latent_embeddings(
                        image=x["image"],
                        text=x["text"],
                    )
                    .detach()
                    .numpy()
                },
                batch_size=5,
                batched=True,
            )
            indexed_external_document.add_faiss_index(
                column="latent_embeddings"
            )
        else:
            indexed_external_document = load_from_disk(
                f"{external_document}/tiny_sketches_with_latent_embeddings"
            )
            indexed_external_document.load_faiss_index(
                "latent_embeddings", f"{external_document}/multimodal.faiss"
            )
        return indexed_external_document


class Generator(nn.Module):
    """
    This component is essentially a decoder that generates both image and text
    """

    def __init__(self):
        super().__init__()

    def forward(self):
        """_summary_"""


if __name__ == "__main__":
    retriever = Retriever()
    query_image = Image.open("tmpidvsx85p.PNG")
    query_text = "The plane is flying in the sky"
    retrieved_documents = retriever(query_image, query_text)
    retriever.save_external_documents("artifacts")
